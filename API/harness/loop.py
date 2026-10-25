"""The agent loop: call the model, run every tool it asked for, send all results back in
ONE user message, repeat while stop_reason == "tool_use"."""
from __future__ import annotations

from shared.prompts import EFFORT, MODEL
from shared.toolkit import Toolkit

from . import caching, tools
from .history import History
from .log import log
from .tokens import TurnMeter

MAX_STEPS = 12


def to_dict(block) -> dict:
    if isinstance(block, dict):
        return block
    return block.to_dict() if hasattr(block, "to_dict") else block.model_dump(exclude_none=True)


class ModelCaller:
    """Thin wrapper over client.messages.create; returns plain dicts."""

    def __init__(self, client, meter: TurnMeter, system_prompt: str):
        self.client, self.meter, self.system_prompt = client, meter, system_prompt

    def __call__(self, messages, label="call", with_system=True, with_tools=True) -> dict:
        kwargs = dict(model=MODEL, max_tokens=16000, thinking={"type": "adaptive"},
                      output_config={"effort": EFFORT}, messages=caching.mark_messages(messages))
        if with_system:
            kwargs["system"] = caching.system_blocks(self.system_prompt)
        if with_tools:
            kwargs["tools"] = tools.schemas()
        log("loop.py", f"{label}: request with {len(messages)} messages")
        resp = self.client.messages.create(**kwargs)
        self.meter.add(resp.usage, label)
        return {"content": [to_dict(b) for b in resp.content], "stop_reason": resp.stop_reason}

    def summarize(self, prompt: str) -> str:
        """One tool-less, system-less request; used by compaction.py."""
        resp = self([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    label="compaction", with_system=False, with_tools=False)
        return "".join(b["text"] for b in resp["content"] if b.get("type") == "text")


def run_turn(call: ModelCaller, history: History, toolkit: Toolkit, user_message: dict) -> dict:
    history.append(user_message)
    used, text = [], ""
    for step in range(1, MAX_STEPS + 1):
        resp = call(history.messages, label=f"step {step}")
        history.append({"role": "assistant", "content": resp["content"]})
        stop = resp["stop_reason"]
        log("loop.py", f"step {step}: stop_reason={stop}")
        text = "".join(b["text"] for b in resp["content"] if b.get("type") == "text")
        if stop == "tool_use":
            calls = [b for b in resp["content"] if b.get("type") == "tool_use"]
            used += [b["name"] for b in calls]
            results = [tools.execute(toolkit, b) for b in calls]
            history.append({"role": "user", "content": results})  # all results, one message
            continue
        if stop == "pause_turn":  # server paused a long turn; resend to let it continue
            continue
        if stop == "refusal":
            text = text or "[the model declined this request]"
        break
    else:
        log("loop.py", f"hit MAX_STEPS={MAX_STEPS}, ending turn")
    return {"text": text, "tools": used}
