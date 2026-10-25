"""What goes into each turn besides the user's words.

The system prompt and tools never change (they are the cached prefix). Per-turn facts
go into a small <context> block at the start of the *user message*, so the history
stays append-only. Included: today's date, relevant memory, and a 3-line deadline
digest instead of the full task list (the model calls list_tasks if it needs more).
"""
from __future__ import annotations

from shared.toolkit import Toolkit

from . import memory
from .log import log

DIGEST_TASKS = 3


def build_user_message(toolkit: Toolkit, user_text: str, today: str) -> dict:
    block = build_context_block(toolkit, user_text, today)
    return {"role": "user", "content": [{"type": "text", "text": block}, {"type": "text", "text": user_text}]}


def build_gemini_user_message(toolkit: Toolkit, user_text: str, today: str) -> dict:
    """Same context, Gemini wire shape."""
    block = build_context_block(toolkit, user_text, today)
    return {"role": "user", "parts": [{"text": block}, {"text": user_text}]}


def build_context_block(toolkit: Toolkit, user_text: str, today: str) -> str:
    lines = [f"today: {today}"]
    mem = memory.relevant(toolkit.memory, user_text)
    lines += mem or ["memory: empty"]
    open_tasks = toolkit.planner.list_tasks("open")
    for t in open_tasks[:DIGEST_TASKS]:
        lines.append(f"due {t['due']} {t['id']} {t['course']} {t['kind']}")
    if len(open_tasks) > DIGEST_TASKS:
        lines.append(f"(+{len(open_tasks) - DIGEST_TASKS} more open tasks: use list_tasks)")
    block = "<context>\n" + "\n".join(lines) + "\n</context>"
    log("context.py", f"context block {len(block)} chars: {len(mem)} memory lines, "
                      f"{min(len(open_tasks), DIGEST_TASKS)}/{len(open_tasks)} tasks")
    return block
