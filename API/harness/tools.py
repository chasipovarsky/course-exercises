"""Tool schemas sent to the model, and dispatch of tool_use blocks to the shared toolkit."""
from __future__ import annotations

import json

from shared.toolkit import SPECS, Toolkit

from .log import log


def schemas() -> list[dict]:
    # sorted + frozen: the tools array is the start of the cached prefix, any reorder is a cache miss
    return [{"name": s["name"], "description": s["description"], "input_schema": s["input_schema"],
             "strict": True} for s in sorted(SPECS, key=lambda s: s["name"])]


def execute(toolkit: Toolkit, block: dict) -> dict:
    """One tool_use block -> one tool_result block. Failures go back to the model as is_error."""
    name, args = block["name"], block.get("input") or {}
    try:
        text, is_error = toolkit.run(name, args), False
    except (ValueError, TypeError) as e:  # bad arguments: the model can fix these itself
        text, is_error = f"error: {e}", True
    except Exception as e:  # network failure etc.: report it, keep the loop alive
        text, is_error = f"error: {type(e).__name__}: {e}", True
    log("tools.py", f"{name}({json.dumps(args, ensure_ascii=False)[:120]}) -> "
                    f"{'ERROR ' if is_error else ''}{len(text)} chars")
    result = {"type": "tool_result", "tool_use_id": block["id"], "content": text}
    if is_error:
        result["is_error"] = True
    return result
