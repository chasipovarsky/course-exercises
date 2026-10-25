"""The agent's tools, defined once and used by both harnesses.

Each tool = name + description + JSON schema + a method that returns a *compact* string.
Compact = one line per record, short ids, no JSON pretty-printing: tool results are
re-sent on every later model call, so every byte here is paid for many times.
Bad input raises ValueError; each harness turns that into an is_error result.
"""
from __future__ import annotations

from .holidays import Holidays
from .memory_store import PREF_KEYS, MemoryStore
from .planner import KINDS, STATUSES, Planner
from .store import Store

MAX_RESULT_CHARS = 1500

SPECS = [
    {"name": "add_task",
     "description": "Add an assignment/exam/project to track. hours = estimated study hours still needed.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["course", "title", "kind", "due", "hours"],
                      "properties": {"course": {"type": "string"}, "title": {"type": "string"},
                                     "kind": {"type": "string", "enum": list(KINDS)},
                                     "due": {"type": "string", "description": "YYYY-MM-DD"},
                                     "hours": {"type": "number"}}}},
    {"name": "list_tasks",
     "description": "List tracked tasks, soonest due first. Output: id|kind|course|title|due|hours|status.",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "properties": {"status": {"type": "string", "enum": [*STATUSES, "all"]},
                                     "course": {"type": "string", "description": "substring filter"}}}},
    {"name": "update_task",
     "description": "Change a task's status (open/done/dropped), due date or remaining hours.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["task_id"],
                      "properties": {"task_id": {"type": "string"},
                                     "status": {"type": "string", "enum": list(STATUSES)},
                                     "due": {"type": "string"}, "hours": {"type": "number"}}}},
    {"name": "plan_week",
     "description": ("Build and save a 7-day study plan from start_date using open tasks, the saved "
                     "preferences (daily_study_hours, off_days) and Israeli holidays (fetched automatically)."),
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["start_date"],
                      "properties": {"start_date": {"type": "string", "description": "YYYY-MM-DD"}}}},
    {"name": "get_holidays",
     "description": "Israeli holidays between two dates from the Hebcal web API. '*' marks a day off (yom tov).",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["start", "end"],
                      "properties": {"start": {"type": "string"}, "end": {"type": "string"}}}},
    {"name": "set_preference",
     "description": ("Save a long-term study preference (remembered across sessions). A new value replaces "
                     "the old one. Keys: " + "; ".join(f"{k} = {v}" for k, v in PREF_KEYS.items())),
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["key", "value"],
                      "properties": {"key": {"type": "string", "enum": list(PREF_KEYS)},
                                     "value": {"type": "string"}}}},
    {"name": "remember",
     "description": "Save a free-form decision or fact worth remembering across sessions (not a preference key).",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["topic", "text"],
                      "properties": {"topic": {"type": "string"}, "text": {"type": "string"}}}},
    {"name": "recall",
     "description": "Search long-term memory: returns current preferences, superseded values and matching notes.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["query"],
                      "properties": {"query": {"type": "string"}}}},
]


def fmt_task(t: dict) -> str:
    return f"{t['id']}|{t['kind']}|{t['course']}|{t['title']}|{t['due']}|{t['hours']:g}h|{t['status']}"


def fmt_plan(p: dict) -> str:
    lines = [f"{p['id']} week {p['start']} (max {p['params']['daily_study_hours']:g}h/day)"]
    for d in p["days"]:
        if "skip" in d:
            lines.append(f"{d['date']} {d['weekday']}: - ({d['skip']})")
        else:
            items = ", ".join(f"{i['task']} {i['hours']:g}h" for i in d["items"]) or "free"
            lines.append(f"{d['date']} {d['weekday']}: {items}")
    if p["unscheduled"]:
        lines.append("left after week: " + ", ".join(f"{k} {v:g}h" for k, v in p["unscheduled"].items()))
    return "\n".join(lines)


def fmt_value(v) -> str:
    return ",".join(v) if isinstance(v, list) else f"{v:g}" if isinstance(v, float) else str(v)


def clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} chars cut]"


class Toolkit:
    def __init__(self, data_dir, clock, fetch=None):
        """clock() -> ISO timestamp string; injectable so runs are reproducible."""
        self.store = Store(data_dir)
        self.memory = MemoryStore(self.store)
        self.holidays = Holidays(self.store, fetch)
        self.planner = Planner(self.store, self.memory, self.holidays)
        self.clock = clock

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            raise ValueError(f"unknown tool {name}")
        return clip(fn(**args))

    def t_add_task(self, course, title, kind, due, hours):
        return "added " + fmt_task(self.planner.add_task(course, title, kind, due, hours))

    def t_list_tasks(self, status="open", course=None):
        rows = self.planner.list_tasks(status, course)
        return "\n".join(fmt_task(t) for t in rows) or f"no {status} tasks"

    def t_update_task(self, task_id, status=None, due=None, hours=None):
        return "updated " + fmt_task(self.planner.update_task(task_id, status, due, hours))

    def t_plan_week(self, start_date):
        return fmt_plan(self.planner.plan_week(start_date, self.clock()))

    def t_get_holidays(self, start, end):
        rows = self.holidays.between(start, end)
        return "\n".join(f"{h['date']} {h['title']}{' *' if h['day_off'] else ''}" for h in rows) or "no holidays"

    def t_set_preference(self, key, value):
        return self.memory.set_preference(key, value, self.clock())

    def t_remember(self, topic, text):
        return self.memory.add_note(topic, text, self.clock())

    def t_recall(self, query):
        prefs = self.memory.preferences()
        lines = [f"pref {k}={fmt_value(v)}" for k, v in prefs.items()]
        lines += [f"old {h['key']}={fmt_value(h['value'])} (replaced {h['superseded_at'][:10]})" for h in self.memory.history()]
        lines += [f"{n['id']} [{n['topic']}] {n['text']}" for n in self.memory.search(query)]
        return "\n".join(lines) or "nothing remembered yet"
