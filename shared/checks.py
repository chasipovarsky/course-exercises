"""Pass/fail checks for a scenario run. Most read the saved state on disk, not the reply text."""
from __future__ import annotations

from .memory_store import MemoryStore
from .planner import Planner
from .store import Store


def _tasks_matching(tasks, course_contains):
    return [t for t in tasks if course_contains in t["course"] or course_contains in t["title"]]


def run_checks(checks: list[dict], data_dir, turns: dict[str, dict]) -> list[dict]:
    store = Store(data_dir)
    memory = MemoryStore(store)
    tasks = store.load(Planner.TASKS, [])
    plans = store.load(Planner.PLANS, [])
    results = []
    for c in checks:
        try:
            ok, detail = CHECKS[c["type"]](c, store=store, memory=memory, tasks=tasks, plans=plans, turns=turns)
        except Exception as e:  # a malformed state is a failed check, not a crash
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"id": c["id"], "about": c.get("about", ""), "passed": ok, "detail": detail})
    return results


def check_task_exists(c, tasks, **_):
    missing = []
    for want in c["tasks"]:
        hits = [t for t in _tasks_matching(tasks, want["course_contains"])
                if t["due"] == want["due"] and ("kind" not in want or t["kind"] == want["kind"])]
        if len(hits) != 1:
            missing.append(f"{want['course_contains']}: {len(hits)} matches")
    return not missing, "; ".join(missing) or f"{len(tasks)} tasks on disk"


def check_task_status(c, tasks, **_):
    hits = _tasks_matching(tasks, c["course_contains"])
    statuses = [t["status"] for t in hits]
    return bool(hits) and all(s == c["status"] for s in statuses), f"statuses={statuses}"


def check_pref_equals(c, memory, **_):
    v = memory.preferences().get(c["key"])
    return v == c["value"], f"{c['key']}={v}"


def check_pref_history(c, memory, **_):
    cur = memory.preferences().get(c["key"])
    old = [h["value"] for h in memory.history() if h["key"] == c["key"]]
    return cur == c["current"] and c["old"] in old, f"current={cur} archived={old}"


def check_plan_constraints(c, plans, tasks, **_):
    if not plans:
        return False, "no plans saved"
    same_start = [p for p in plans if p["start"] == c["start"]]
    if not same_start:
        return False, f"no plan starting {c['start']} (have {[p['start'] for p in plans]})"
    p = same_start[c["plan"]]  # 0 = first plan for that week, -1 = latest re-plan
    problems = []
    done = {t["id"] for t in tasks if t["status"] == "done"}
    total = 0.0
    for d in p["days"]:
        hours = sum(i["hours"] for i in d["items"])
        total += hours
        if hours > c["max_daily"]:
            problems.append(f"{d['date']} {hours}h")
        if hours and (d["weekday"] in c["no_weekdays"] or d["date"] in c["no_dates"]):
            problems.append(f"study on {d['date']} {d['weekday']}")
        if c.get("no_tasks_done") and any(i["task"] in done for i in d["items"]):
            problems.append(f"done task planned on {d['date']}")
    if total == 0:
        problems.append("empty plan")
    return not problems, "; ".join(problems) or f"{p['id']}: {total:g}h planned, all constraints hold"


def check_holidays_cached(c, store, **_):
    files = list((store.dir / "cache").glob("hebcal_*.json"))
    return bool(files), f"{len(files)} cached Hebcal responses"


def check_response_contains(c, turns, **_):
    text = turns.get(c["turn"], {}).get("response", "")
    hit = [s for s in c["any_of"] if s in text]
    return bool(hit), f"found {hit}" if hit else f"none of {c['any_of']} in reply"


CHECKS = {
    "task_exists": check_task_exists, "task_status": check_task_status,
    "pref_equals": check_pref_equals, "pref_history": check_pref_history,
    "plan_constraints": check_plan_constraints, "holidays_cached": check_holidays_cached,
    "response_contains": check_response_contains,
}
