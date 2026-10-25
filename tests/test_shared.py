"""Domain tools, planner and scenario checks: no model involved."""
import json

import pytest
from conftest import ROOT, fake_fetch

from shared.checks import run_checks
from shared.memory_store import normalize_days
from shared.toolkit import Toolkit


@pytest.fixture
def tk(data_dir):
    return Toolkit(data_dir, clock=lambda: "2027-04-12T10:00:00", fetch=fake_fetch)


def test_hebrew_and_english_weekdays():
    assert normalize_days("שישי ושבת") == ["Fri", "Sat"]
    assert normalize_days("Saturday, fri") == ["Fri", "Sat"]
    assert normalize_days("יום ראשון") == ["Sun"]


def test_plan_respects_prefs_holidays_and_due_dates(tk):
    tk.run("add_task", {"course": "OS", "title": "HW4", "kind": "assignment", "due": "2027-04-20", "hours": 5})
    tk.run("add_task", {"course": "Algo", "title": "Final", "kind": "exam", "due": "2027-04-29", "hours": 12})
    tk.run("set_preference", {"key": "daily_study_hours", "value": "3"})
    tk.run("set_preference", {"key": "off_days", "value": "Fri,Sat"})
    out = tk.run("plan_week", {"start_date": "2027-04-18"})
    plan = tk.planner.plans()[-1]
    by_date = {d["date"]: d for d in plan["days"]}
    assert by_date["2027-04-22"]["skip"] == "Pesach I"
    assert by_date["2027-04-23"]["skip"] == "off day"
    assert all(sum(i["hours"] for i in d["items"]) <= 3 for d in plan["days"])
    os_days = [d["date"] for d in plan["days"] for i in d["items"] if i["task"] == "T1"]
    assert max(os_days) < "2027-04-20"  # never planned on/after its due date
    assert "Pesach I" in out and len(out) < 600  # compact


def test_external_tool_is_cached(tk):
    tk.run("get_holidays", {"start": "2027-04-18", "end": "2027-04-24"})
    tk.run("get_holidays", {"start": "2027-04-18", "end": "2027-04-24"})
    assert tk.holidays.network_calls == 1
    assert "Pesach I *" in tk.run("get_holidays", {"start": "2027-04-18", "end": "2027-04-24"})


def test_bad_input_raises_value_error(tk):
    with pytest.raises(ValueError):
        tk.run("add_task", {"course": "x", "title": "y", "kind": "party", "due": "2027-01-01", "hours": 1})
    with pytest.raises(ValueError):
        tk.run("set_preference", {"key": "daily_study_hours", "value": "a lot"})
    with pytest.raises(ValueError):
        tk.run("plan_week", {"start_date": "next week"})


def test_scenario_checks_pass_on_expected_state(tk, data_dir):
    """Replays the scenario's intended tool effects and runs the real scenario.json checks."""
    tk.run("add_task", {"course": "אלגוריתמים", "title": "מבחן מועד א'", "kind": "exam", "due": "2027-04-29", "hours": 12})
    tk.run("add_task", {"course": "מערכות הפעלה", "title": "תרגיל 4", "kind": "assignment", "due": "2027-04-20", "hours": 5})
    tk.run("set_preference", {"key": "daily_study_hours", "value": "3"})
    tk.run("set_preference", {"key": "off_days", "value": "שישי ושבת"})
    tk.run("plan_week", {"start_date": "2027-04-18"})
    tk.run("update_task", {"task_id": "T2", "status": "done"})
    tk.run("set_preference", {"key": "daily_study_hours", "value": "2"})
    tk.run("plan_week", {"start_date": "2027-04-25"})
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    results = run_checks(scenario["checks"], data_dir, {"t5": {"response": "אמרת 3 שעות ביום"}})
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed
    assert len(results) >= 5


def test_scenario_run_starts_from_clean_state(tmp_path, data_dir):
    import asyncio
    from shared.runner import run_scenario
    data_dir.mkdir(parents=True)
    (data_dir / "memory.json").write_text('{"preferences": {"daily_study_hours": {"value": 9}}}', encoding="utf-8")
    (data_dir / "sessions").mkdir()
    seen = []

    class Dummy:
        def __init__(self):
            seen.append(sorted(p.name for p in data_dir.iterdir()))

        def turn(self, text):
            return {"response": "", "tools": [], "input_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
                    "output_tokens": 0, "model_calls": 0, "cost_usd": 0.0, "seconds": 0.0}

    asyncio.run(run_scenario(lambda *a: Dummy(), str(ROOT / "scenario.json"), str(tmp_path / "out.json"),
                             data_dir, "test", {}))
    assert seen[0] == []  # memory.json and sessions/ from the "previous run" are gone


def test_checks_fail_when_contradiction_not_saved(tk, data_dir):
    tk.run("set_preference", {"key": "daily_study_hours", "value": "3"})
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    results = {r["id"]: r for r in run_checks(scenario["checks"], data_dir, {})}
    assert not results["contradiction_resolved"]["passed"]
    assert not results["plan1_respects_holiday"]["passed"]
