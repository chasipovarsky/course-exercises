"""The agent loop on Gemini: call the model, run every function_call it returned, send all
function_response parts back in ONE user turn, repeat while the model returns function calls.

Model turns are stored exactly as received (including thought_signature, base64 in JSON),
because Gemini needs them back unchanged to keep its reasoning across tool calls.
Free tier is rate-limited, so 429 (and 503 overload) is retried with exponential backoff.
"""
from __future__ import annotations

import re
import time

from google.genai import errors, types

from shared.toolkit import Toolkit

from . import gemini_caching, gemini_tools
from .gemini_tokens import MODEL, GeminiTurnMeter
from .history import History
from .log import log

MAX_STEPS = 12
RETRY_CODES = (429, 503)
MAX_ATTEMPTS = 6
BASE_DELAY, MAX_DELAY = 5.0, 60.0


def _retry_delay(e: Exception, attempt: int) -> float:
    """Server-suggested delay (RetryInfo.retryDelay, e.g. '17s') if present, else exponential backoff."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(getattr(e, "details", "")) + str(e))
    if m:
        return min(float(m.group(1)) + 1.0, MAX_DELAY)
    return min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)


class GeminiCaller:
    """Thin wrapper over client.models.generate_content; returns plain dicts."""

    def __init__(self, client, meter: GeminiTurnMeter, config, sleep=time.sleep):
        self.client, self.meter, self.config, self.sleep = client, meter, config, sleep

    def _generate(self, contents, config, label):
        # stored dicts -> Content; base64 thought_signature strings decode back to the original bytes
        contents = [types.Content.model_validate(c) for c in contents]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(model=MODEL, contents=contents, config=config)
            except (errors.ClientError, errors.ServerError) as e:
                if e.code not in RETRY_CODES or attempt == MAX_ATTEMPTS:
                    raise
                delay = _retry_delay(e, attempt)
                log("gemini_loop.py", f"{label}: HTTP {e.code}, retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s")
                self.sleep(delay)

    def __call__(self, contents, label="call", config=None) -> dict:
        log("gemini_loop.py", f"{label}: request with {len(contents)} contents")
        resp = self._generate(contents, config or self.config, label)
        self.meter.add(resp.usage_metadata, label)
        gemini_caching.report(resp.usage_metadata, label)
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None or not cand.content.parts:
            reason = getattr(cand, "finish_reason", None) or getattr(resp, "prompt_feedback", None)
            return {"content": None, "finish_reason": str(reason)}
        return {"content": cand.content.model_dump(mode="json", exclude_none=True),
                "finish_reason": str(cand.finish_reason)}

    def summarize(self, prompt: str) -> str:
        """One tool-less, system-less request; used by compaction.py."""
        resp = self([{"role": "user", "parts": [{"text": prompt}]}], label="compaction",
                    config=gemini_caching.summary_config())
        return _text(resp["content"])


def _text(content: dict | None) -> str:
    if not content:
        return ""
    return "".join(p["text"] for p in content.get("parts", []) if "text" in p and not p.get("thought"))


def run_turn(call: GeminiCaller, history: History, toolkit: Toolkit, user_message: dict) -> dict:
    history.append(user_message)
    used, text = [], ""
    for step in range(1, MAX_STEPS + 1):
        resp = call(history.messages, label=f"step {step}")
        content = resp["content"]
        if content is None:  # blocked or empty candidate: nothing to append, end the turn
            log("gemini_loop.py", f"step {step}: empty response ({resp['finish_reason']})")
            text = text or f"[no response from the model: {resp['finish_reason']}]"
            break
        history.append(content)
        calls = [p["function_call"] for p in content["parts"] if "function_call" in p]
        log("gemini_loop.py", f"step {step}: {len(calls)} function_call(s), finish_reason={resp['finish_reason']}")
        text = _text(content) or text
        if calls:
            used += [c["name"] for c in calls]
            parts = [gemini_tools.execute(toolkit, c) for c in calls]
            history.append({"role": "user", "parts": parts})  # all responses, one turn
            continue
        break
    else:
        log("gemini_loop.py", f"hit MAX_STEPS={MAX_STEPS}, ending turn")
    return {"text": text, "tools": used}
