"""Where cache_control breakpoints go (max 4 per request; we use 2).

1. end of the system prompt -> caches tools + system (identical on every call)
2. last block of the last message -> caches the whole conversation so far, so the next
   call in the tool loop (or the next turn) reads it back at 0.05x price.
Markers are added to a copy at request time; stored history never carries them.
"""
from __future__ import annotations

import copy

from .log import log

EPHEMERAL = {"type": "ephemeral"}


def system_blocks(system_prompt: str) -> list[dict]:
    return [{"type": "text", "text": system_prompt, "cache_control": EPHEMERAL}]


def mark_messages(messages: list[dict]) -> list[dict]:
    out = copy.deepcopy(messages)
    for block in reversed(out[-1]["content"]):
        if block.get("type") not in ("thinking", "redacted_thinking"):  # thinking blocks can't carry markers
            block["cache_control"] = EPHEMERAL
            break
    log("caching.py", f"breakpoints: system + message[{len(out) - 1}]")
    return out
