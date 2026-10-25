"""Conversation history: an append-only JSONL file per session.

Records are {"type": "message", "message": {...}} or {"type": "compaction", "summary": ...}.
Nothing is ever rewritten; the working message list is rebuilt from the last compaction on.
Shared by both providers; only the wire shape of a message differs
(Anthropic: {"role", "content": [blocks]}, Gemini: {"role", "parts": [parts]}).
"""
from __future__ import annotations

import json
from pathlib import Path

from .log import log


class History:
    def __init__(self, sessions_dir: str | Path, session_id: str, provider: str = "anthropic"):
        self.path = Path(sessions_dir) / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.summary_message = SUMMARY_MESSAGE[provider]
        self.messages: list[dict] = []
        if self.path.exists():
            self._replay()

    def _replay(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["type"] == "compaction":
                self.messages = [self.summary_message(rec["summary"])]
            else:
                self.messages.append(rec["message"])
        log("history.py", f"resumed {self.session_id}: {len(self.messages)} messages from {self.path.name}")

    def _write(self, rec: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append(self, message: dict) -> None:
        self.messages.append(message)
        self._write({"type": "message", "message": message})
        blocks = message.get("content", message.get("parts", []))
        log("history.py", f"append {message['role']} ({len(blocks)} blocks) -> {self.path.name}")

    def compact(self, summary: str) -> None:
        self._write({"type": "compaction", "summary": summary})
        self.messages = [self.summary_message(summary)]
        log("history.py", "compaction record appended; working history reset to summary")


def _summary_text(summary: str) -> str:
    return f"<conversation_summary>\n{summary}\n</conversation_summary>"


def summary_message(summary: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": _summary_text(summary)}]}


def gemini_summary_message(summary: str) -> dict:
    return {"role": "user", "parts": [{"text": _summary_text(summary)}]}


SUMMARY_MESSAGE = {"anthropic": summary_message, "gemini": gemini_summary_message}
