"""Long-term memory: structured preferences + free-form notes, persisted in memory.json.

Preferences are keyed, so a new value for an existing key *replaces* it and the old
value is kept in `history` with a superseded timestamp (that is how contradictions
are resolved: latest statement wins, nothing is silently lost).
"""
from __future__ import annotations

import re

from .store import Store

PREF_KEYS = {
    "daily_study_hours": "max study hours per day (number)",
    "off_days": "weekdays with no study, e.g. 'Fri,Sat'",
    "session_length_hours": "preferred length of one study block (number)",
    "preferred_time": "morning / afternoon / evening",
}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAY_ALIASES = {
    "mon": "Mon", "monday": "Mon", "שני": "Mon",
    "tue": "Tue", "tuesday": "Tue", "שלישי": "Tue",
    "wed": "Wed", "wednesday": "Wed", "רביעי": "Wed",
    "thu": "Thu", "thursday": "Thu", "חמישי": "Thu",
    "fri": "Fri", "friday": "Fri", "שישי": "Fri",
    "sat": "Sat", "saturday": "Sat", "שבת": "Sat",
    "sun": "Sun", "sunday": "Sun", "ראשון": "Sun",
}


def normalize_days(value: str) -> list[str]:
    out = []
    for tok in re.split(r"[,\s/]+", value.strip()):
        tok = tok.strip().lower()
        if tok not in _DAY_ALIASES and tok.startswith("ו"):  # Hebrew "and": ושבת -> שבת
            tok = tok[1:]
        if tok in _DAY_ALIASES and _DAY_ALIASES[tok] not in out:
            out.append(_DAY_ALIASES[tok])
    if not out:
        raise ValueError(f"no weekday names found in {value!r}; use e.g. 'Fri,Sat'")
    return sorted(out, key=WEEKDAYS.index)


def normalize_pref(key: str, value: str):
    if key not in PREF_KEYS:
        raise ValueError(f"unknown preference {key!r}; allowed: {', '.join(PREF_KEYS)}")
    if key in ("daily_study_hours", "session_length_hours"):
        m = re.search(r"\d+(\.\d+)?", str(value))
        if not m:
            raise ValueError(f"{key} needs a number, got {value!r}")
        num = float(m.group())
        if not 0 < num <= 16:
            raise ValueError(f"{key}={num} is out of range (0-16]")
        return num
    if key == "off_days":
        return normalize_days(str(value))
    return str(value).strip()


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 1}


class MemoryStore:
    FILE = "memory.json"

    def __init__(self, store: Store):
        self.store = store

    def _data(self) -> dict:
        return self.store.load(self.FILE, {"preferences": {}, "notes": [], "history": []})

    def preferences(self) -> dict:
        return {k: v["value"] for k, v in self._data()["preferences"].items()}

    def notes(self) -> list[dict]:
        return self._data()["notes"]

    def history(self) -> list[dict]:
        return self._data()["history"]

    def set_preference(self, key: str, value: str, now: str) -> str:
        norm = normalize_pref(key, value)
        data = self._data()
        old = data["preferences"].get(key)
        if old and old["value"] == norm:
            return f"unchanged {key}={norm}"
        if old:
            data["history"].append({"key": key, "value": old["value"], "set_at": old["set_at"],
                                    "superseded_at": now, "replaced_by": norm})
        data["preferences"][key] = {"value": norm, "set_at": now}
        self.store.save(self.FILE, data)
        return f"updated {key}: {old['value']} -> {norm} (old value archived)" if old else f"saved {key}={norm}"

    def add_note(self, topic: str, text: str, now: str) -> str:
        data = self._data()
        nid = f"N{len(data['notes']) + 1}"
        data["notes"].append({"id": nid, "topic": topic, "text": text, "at": now})
        self.store.save(self.FILE, data)
        return f"saved note {nid}"

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Keyword-overlap ranking over notes; cheap, deterministic, no embeddings needed."""
        q = _words(query)
        scored = []
        for n in self.notes():
            s = len(q & _words(n["topic"] + " " + n["text"]))
            if s:
                scored.append((s, n["at"], n))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)  # best match, then newest
        return [n for _, _, n in scored[:k]]
