"""Run the API-harness agent on a scenario, or chat with it.

  python API/main.py --scenario scenario.json --out results/api.json
  python API/main.py --provider gemini --scenario scenario.json --out results/api_gemini.json
  python API/main.py --chat [--session my-session]     (resumes if that session exists)

Keys come only from the environment (never in code, logs or results):
anthropic -> ANTHROPIC_API_KEY, gemini -> GEMINI_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from shared.runner import ROOT, count_loc, run_scenario, utf8_console  # noqa: E402

DATA_DIR = {"anthropic": ROOT / "runs" / "api", "gemini": ROOT / "runs" / "api_gemini"}
DEFAULT_OUT = {"anthropic": "results/api.json", "gemini": "results/api_gemini.json"}
PARTS = ["loop", "tools", "history", "memory", "context", "compaction", "caching", "tokens"]
GEMINI_FILES = {"loop": "gemini_loop", "tools": "gemini_tools", "caching": "gemini_caching", "tokens": "gemini_tokens"}


def loc(provider: str = "anthropic") -> dict:
    files = GEMINI_FILES if provider == "gemini" else {}
    out = {p: count_loc(HERE / "harness" / f"{files.get(p, p)}.py") for p in PARTS}
    out["(log helper)"] = count_loc(HERE / "harness" / "log.py")
    agent_file = "gemini_agent.py" if provider == "gemini" else "agent.py"
    out[f"{agent_file} + main.py"] = count_loc(HERE / agent_file) + count_loc(HERE / "main.py")
    out["total"] = sum(out.values())
    return out


def gemini_client():
    """Reads GEMINI_API_KEY only (not GOOGLE_API_KEY) and passes it straight to the client."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_API_KEY is not set. In cmd: set GEMINI_API_KEY=...")
    from google import genai
    return genai.Client(api_key=key)


def make_factory(provider: str, client, data_dir: Path):
    if provider == "gemini":
        from gemini_agent import GeminiAgent
        return lambda session_id, today, threshold: GeminiAgent(client, data_dir, session_id, today, threshold)
    from agent import ApiAgent
    return lambda session_id, today, threshold: ApiAgent(client, data_dir, session_id, today, threshold)


def main() -> None:
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--scenario")
    ap.add_argument("--out")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--session", default="chat")
    args = ap.parse_args()
    if not (args.scenario or args.chat):
        ap.print_help()
        return
    if args.provider == "gemini":
        client = gemini_client()
    else:
        import anthropic
        client = anthropic.Anthropic()

    if args.scenario:
        make = make_factory(args.provider, client, DATA_DIR[args.provider])
        meta = None
        if args.provider == "gemini":
            from harness.gemini_tokens import MODEL, PRICE, PRICE_SOURCE, THINKING_LEVEL
            meta = {"provider": "gemini", "model": MODEL, "effort": f"thinking_level={THINKING_LEVEL}",
                    "pricing_usd_per_mtok": PRICE, "pricing_source": PRICE_SOURCE}
        asyncio.run(run_scenario(make, args.scenario, args.out or DEFAULT_OUT[args.provider],
                                 DATA_DIR[args.provider], "api", loc(args.provider), meta))
    else:
        make = make_factory(args.provider, client, ROOT / "runs" / f"chat-{args.provider}")
        agent = make(args.session, date.today().isoformat(), 30000)
        while (text := input("\nyou> ").strip()) not in ("", "exit"):
            r = agent.turn(text)
            print(f"agent> {r['response']}\n  [{r['model_calls']} calls, ${r['cost_usd']:.4f}]")


if __name__ == "__main__":
    main()
