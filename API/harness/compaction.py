"""Context-window management: when the last prompt passed a threshold, replace the whole
conversation with a model-written summary ("simple compaction").

Runs only between user turns, and replaces *everything* (no kept tail), so no thinking
block is ever replayed after its prefix changed. Long-term facts are safe anyway: they
live in memory.json / tasks.json, not in the transcript.
"""
from __future__ import annotations

import json

from .history import History
from .log import log

SUMMARY_PROMPT = (
    "Summarize the transcript inside <summary></summary> tags so the conversation can continue in a new "
    "context window without redoing work. Preserve exactly: what the user asked for, decided or stated as "
    "a preference; task ids, dates and numbers; what has been completed; and what is still open. Keep the "
    "user's words close to the original; condense your own explanations. Respond with text only."
)


def render(messages: list[dict]) -> str:
    """Transcript as plain text for the summarizer. Thinking / thought parts are left out."""
    out = []
    for m in messages:
        if "parts" in m:  # Gemini shape
            for p in m["parts"]:
                if p.get("thought"):
                    continue
                if "text" in p:
                    out.append(f"{m['role'].upper()}: {p['text']}")
                elif "function_call" in p:
                    fc = p["function_call"]
                    out.append(f"TOOL CALL {fc['name']} {json.dumps(fc.get('args'), ensure_ascii=False)}")
                elif "function_response" in p:
                    out.append(f"TOOL RESULT: {json.dumps(p['function_response'].get('response'), ensure_ascii=False)}")
            continue
        for b in m["content"]:
            t = b.get("type")
            if t == "text":
                out.append(f"{m['role'].upper()}: {b['text']}")
            elif t == "tool_use":
                out.append(f"TOOL CALL {b['name']} {json.dumps(b.get('input'), ensure_ascii=False)}")
            elif t == "tool_result":
                out.append(f"TOOL RESULT: {b.get('content')}")
    return "\n".join(out)


def maybe_compact(call_model, history: History, meter, threshold: int) -> bool:
    """call_model is a provider caller (loop.ModelCaller / gemini_loop.GeminiCaller) with .summarize(prompt)."""
    size = meter.last_context_tokens
    if size < threshold or len(history.messages) < 4:
        log("compaction.py", f"context {size} tokens < threshold {threshold}: no compaction")
        return False
    log("compaction.py", f"context {size} tokens >= {threshold}: summarizing {len(history.messages)} messages")
    text = call_model.summarize("<transcript>\n" + render(history.messages) + "\n</transcript>\n\n" + SUMMARY_PROMPT)
    if "<summary>" in text:
        text = text.split("<summary>", 1)[1].split("</summary>", 1)[0]
    history.compact(text.strip())
    return True
