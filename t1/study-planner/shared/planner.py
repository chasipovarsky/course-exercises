"""Domain logic: tasks (assignments / exams) and weekly study plans."""
from __future__ import annotations

from datetime import date, timedelta

from .holidays import Holidays
from .memory_store import WEEKDAYS, MemoryStore
from .store import Store

KINDS = ("exam", "assignment", "project", "reading")
STATUSES = ("open", "done", "dropped")
DEFAULT_DAILY_HOURS = 3.0


def _iso(d: str, field: str) -> str:
    try:
        return date.fromisoformat(d).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {d!r}")


class Planner:
    TASKS = "tasks.json"
    PLANS = "plans.json"

    def __init__(self, store: Store, memory: MemoryStore, holidays: Holidays):
        self.store, self.memory, self.holidays = store, memory, holidays

    # ---- tasks ---------------------------------------------------------
    def tasks(self) -> list[dict]:
        return self.store.load(self.TASKS, [])

    def add_task(self, course: str, title: str, kind: str, due: str, hours: float) -> dict:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not 0 < float(hours) <= 200:
            raise ValueError("hours must be in (0, 200]")
        tasks = self.tasks()
        for t in tasks:  # idempotent: the model sometimes retries an add
            if t["course"] == course and t["title"] == title and t["status"] == "open":
                raise ValueError(f"task already exists as {t['id']}; use update_task")
        task = {"id": f"T{len(tasks) + 1}", "course": course.strip(), "title": title.strip(),
                "kind": kind, "due": _iso(due, "due"), "hours": float(hours), "status": "open"}
        tasks.append(task)
        self.store.save(self.TASKS, tasks)
        return task

    def update_task(self, task_id: str, status=None, due=None, hours=None) -> dict:
        tasks = self.tasks()
        task = next((t for t in tasks if t["id"] == task_id), None)
        if task is None:
            raise ValueError(f"no task {task_id}; call list_tasks to see ids")
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {STATUSES}")
            task["status"] = status
        if due is not None:
            task["due"] = _iso(due, "due")
        if hours is not None:
            task["hours"] = float(hours)
        self.store.save(self.TASKS, tasks)
        return task

    def list_tasks(self, status: str = "open", course: str | None = None) -> list[dict]:
        out = [t for t in self.tasks() if status == "all" or t["status"] == status]
        if course:
            out = [t for t in out if course.lower() in t["course"].lower()]
        return sorted(out, key=lambda t: t["due"])

    # ---- planning ------------------------------------------------------
    def plans(self) -> list[dict]:
        return self.store.load(self.PLANS, [])

    def plan_week(self, start: str, created_at: str) -> dict:
        """Earliest-deadline-first allocation of open tasks onto available days.

        Available = not an off_day preference, not a yom tov (Hebcal), before the task's due date.
        """
        start = _iso(start, "start_date")
        prefs = self.memory.preferences()
        daily = float(prefs.get("daily_study_hours", DEFAULT_DAILY_HOURS))
        off_days = prefs.get("off_days", [])
        s = date.fromisoformat(start)
        days = [s + timedelta(i) for i in range(7)]
        hol = {h["date"]: h["title"] for h in self.holidays.between(days[0].isoformat(), days[-1].isoformat())
               if h["day_off"]}

        remaining = {t["id"]: t["hours"] for t in self.list_tasks("open")}
        tasks = self.list_tasks("open")
        plan_days = []
        for d in days:
            iso, wd = d.isoformat(), WEEKDAYS[d.weekday()]
            entry = {"date": iso, "weekday": wd, "items": []}
            if wd in off_days:
                entry["skip"] = "off day"
            elif iso in hol:
                entry["skip"] = hol[iso]
            else:
                cap = daily
                for t in tasks:
                    if cap <= 0:
                        break
                    if remaining[t["id"]] > 0 and iso < t["due"]:
                        h = min(cap, remaining[t["id"]])
                        entry["items"].append({"task": t["id"], "hours": h})
                        remaining[t["id"]] -= h
                        cap -= h
            plan_days.append(entry)

        unscheduled = {tid: h for tid, h in remaining.items() if h > 0}
        plan = {"id": f"P{len(self.plans()) + 1}", "start": start, "created_at": created_at,
                "params": {"daily_study_hours": daily, "off_days": off_days, "holidays": hol},
                "days": plan_days, "unscheduled": unscheduled}
        plans = self.plans()
        plans.append(plan)
        self.store.save(self.PLANS, plans)
        return plan
