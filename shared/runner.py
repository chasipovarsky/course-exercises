"""Scenario driver shared by API/main.py and SDK/main.py, so both are measured identically."""
from __future__ import annotations

import inspect
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .checks import run_checks
from .prompts import EFFORT, MODEL

ROOT = Path(__file__).resolve().parent.parent
TOTAL_KEYS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "model_calls", "cost_usd",
              "seconds")


def utf8_console() -> None:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")


def clean_dir(path: Path) -> None:
    """Every run starts from a clean state: memory, history (sessions/), tasks, plans, caches."""
    if path.exists():
        old = sorted(p.name for p in path.iterdir())
        shutil.rmtree(path)
        print(f"clean state: deleted {path} ({', '.join(old) or 'empty'})", flush=True)
    path.mkdir(parents=True)


def count_loc(path: Path) -> int:
    """Non-blank lines that are not pure comments/docstring lines."""
    n, in_doc = 0, False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if in_doc:
            in_doc = '"""' not in s
            continue
        if s.startswith('"""'):
            in_doc = s.count('"""') == 1
            continue
        if not s or s.startswith("#"):
            continue
        n += 1
    return n


async def run_scenario(make_agent, scenario_path: str, out_path: str, data_dir: Path, harness: str,
                       loc: dict, meta: dict | None = None) -> dict:
    """meta overrides/extends the header fields (e.g. model and pricing for the Gemini provider)."""
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    clean_dir(data_dir)
    turns, by_id = [], {}
    for session in scenario["sessions"]:
        print(f"\n===== session {session['id']} (new agent instance) =====", flush=True)
        agent = make_agent(session["id"], scenario["today"], scenario.get("compaction_threshold", 30000))
        try:
            for t in session["turns"]:
                print(f"\n>>> [{t['id']}] {t['prompt']}", flush=True)
                r = agent.turn(t["prompt"])
                if inspect.isawaitable(r):  # SDK agent is async, API agent is sync
                    r = await r
                print(f"<<< {r['response']}\n    tools={r['tools']} calls={r['model_calls']} ${r['cost_usd']:.4f}",
                      flush=True)
                rec = {"session": session["id"], "turn": t["id"], "prompt": t["prompt"], **r}
                turns.append(rec)
                by_id[t["id"]] = rec
        finally:
            closing = getattr(agent, "close", None)
            if closing and inspect.isawaitable(c := closing()):
                await c

    checks = run_checks(scenario["checks"], data_dir, by_id)
    totals = {k: round(sum(t[k] for t in turns), 5) for k in TOTAL_KEYS}
    if turns and all("theoretical_cost_usd" in t for t in turns):
        totals["theoretical_cost_usd"] = round(sum(t["theoretical_cost_usd"] for t in turns), 6)
    total_in = totals["input_tokens"] + totals["cache_read_tokens"] + totals["cache_write_tokens"]
    totals["cache_hit_pct"] = round(100 * totals["cache_read_tokens"] / total_in, 1) if total_in else 0.0
    result = {
        "harness": harness, "model": MODEL, "effort": EFFORT, **(meta or {}),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": str(scenario_path), "turns": turns, "totals": totals,
        "checks": checks, "checks_passed": f"{sum(c['passed'] for c in checks)}/{len(checks)}",
        "lines_of_code": loc,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nchecks {result['checks_passed']}:")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']}: {c['detail']}")
    print(f"totals: {totals}\nwritten {out_path}")
    return result
