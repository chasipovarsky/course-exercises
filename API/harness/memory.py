"""Long-term memory: where it lives and what gets pulled out of it each turn.

Stored in <data_dir>/memory.json (shared.memory_store). Retrieval policy:
- current preferences: always (tiny, and plan_week depends on them)
- notes: only the top-k that share words with the user's message
- superseded values: never injected; the model can ask for them with `recall`
"""
from __future__ import annotations

from shared.memory_store import MemoryStore
from shared.toolkit import fmt_value

from .log import log

TOP_K_NOTES = 3


def relevant(memory: MemoryStore, user_text: str) -> list[str]:
    lines = [f"pref {k}={fmt_value(v)}" for k, v in memory.preferences().items()]
    notes = memory.search(user_text, TOP_K_NOTES)
    lines += [f"note {n['id']} [{n['topic']}] {n['text']}" for n in notes]
    total_notes = len(memory.notes())
    log("memory.py", f"retrieved {len(lines) - len(notes)} prefs + {len(notes)}/{total_notes} notes")
    return lines
