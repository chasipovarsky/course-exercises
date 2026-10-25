"""Run the Agent-SDK agent on a scenario, or chat with it.

  python SDK/main.py --scenario scenario.json --out results/sdk.json
  python SDK/main.py --chat [--resume <session_id>]

Runs on your Claude Code login. ANTHROPIC_API_KEY is removed from the environment so
billing never silently moves to an API key.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from agent import SdkAgent  # noqa: E402
from shared.runner import ROOT, count_loc, run_scenario, utf8_console  # noqa: E402

DATA_DIR = ROOT / "runs" / "sdk"
PARTS = ["loop", "tools", "history", "memory", "context", "compaction", "caching", "tokens"]


def loc() -> dict:
    """Lines tagged '# replaces: <part>' per harness part, plus the SDK files' total."""
    out = dict.fromkeys(PARTS, 0)
    for f in ("agent.py", "tools.py"):
        for line in (HERE / f).read_text(encoding="utf-8").splitlines():
            m = re.search(r"# replaces: (.+)", line)
            if m:
                for part in re.findall(r"(\w+)\.py", m.group(1)):
                    out[part] += 1
    out["agent.py + tools.py + main.py"] = sum(count_loc(HERE / f) for f in ("agent.py", "tools.py", "main.py"))
    return out


def clean_claude_code_sessions(cwd: Path) -> None:
    """Claude Code keeps its own transcripts in ~/.claude/projects/<cwd with non-alphanumerics as '-'>.
    Delete only the folder that belongs to this agent's scenario cwd, so a run starts clean."""
    projects = Path.home() / ".claude" / "projects"
    folder = projects / re.sub(r"[^A-Za-z0-9]", "-", str(cwd.resolve()))
    if folder.parent == projects and folder.name.endswith("-runs-sdk") and folder.is_dir():
        n = len(list(folder.iterdir()))
        shutil.rmtree(folder)
        print(f"clean state: deleted {n} Claude Code session files in {folder}", flush=True)


async def chat(resume: str | None) -> None:
    agent = SdkAgent(ROOT / "runs" / "chat-sdk", "chat", date.today().isoformat(), resume=resume)
    try:
        while (text := input("\nyou> ").strip()) not in ("", "exit"):
            r = await agent.turn(text)
            print(f"agent> {r['response']}\n  [{r['model_calls']} calls, ${r['cost_usd']:.4f}, "
                  f"session {r['session_id']}]")
    finally:
        await agent.close()


def main() -> None:
    utf8_console()
    if os.environ.pop("ANTHROPIC_API_KEY", None):
        print("note: ignoring ANTHROPIC_API_KEY so the SDK runs on the Claude Code login", file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--out", default="results/sdk.json")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--resume")
    args = ap.parse_args()

    if args.scenario:
        def make(session_id, today, _threshold):  # compaction threshold is Claude Code's business
            return SdkAgent(DATA_DIR, session_id, today)
        clean_claude_code_sessions(DATA_DIR)  # runs/sdk itself is wiped by run_scenario
        asyncio.run(run_scenario(make, args.scenario, args.out, DATA_DIR, "sdk", loc()))
    elif args.chat:
        asyncio.run(chat(args.resume))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
