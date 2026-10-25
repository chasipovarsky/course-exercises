"""Print the REPORT.md comparison tables from results/*.json.

  python compare.py                   # every result present: api, api_gemini, sdk
  python compare.py api_gemini sdk    # just these two
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARTS = ["loop", "tools", "history", "memory", "context", "compaction", "caching", "tokens"]
DEFAULT = ["api", "api_gemini", "sdk"]


def load(name):
    p = ROOT / "results" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def label(name, r):
    return f"{name} ({r.get('model', '?')})"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names = sys.argv[1:] or DEFAULT
    runs = [(label(n, r), r) for n in names if (r := load(n))]
    if not runs:
        sys.exit("no results found in results/")

    rows = [("input (uncached)", "input_tokens"), ("cache read", "cache_read_tokens"),
            ("cache write", "cache_write_tokens"), ("output", "output_tokens"), ("cache hit %", "cache_hit_pct"),
            ("model calls", "model_calls"), ("cost USD (actual)", "cost_usd"),
            ("cost USD (theoretical, paid tier)", "theoretical_cost_usd"), ("seconds", "seconds")]
    print("| metric | " + " | ".join(n for n, _ in runs) + " |\n|---|" + "---|" * len(runs))
    for text, key in rows:
        print(f"| {text} | " + " | ".join(str(r["totals"].get(key, "-")) for _, r in runs) + " |")
    print("| checks passed | " + " | ".join(r["checks_passed"] for _, r in runs) + " |")

    print("\n| turn | " + " | ".join(f"{n} in/cache_r/cache_w/out" for n, _ in runs)
          + " | " + " | ".join(f"{n} calls, tools" for n, _ in runs) + " |")
    print("|---|" + "---|" * (2 * len(runs)))
    for i, t in enumerate(runs[0][1]["turns"]):
        tok, calls = [], []
        for _, r in runs:
            u = r["turns"][i] if i < len(r["turns"]) else None
            tok.append(f"{u['input_tokens']}/{u['cache_read_tokens']}/{u['cache_write_tokens']}/{u['output_tokens']}"
                       if u else "-")
            calls.append(f"{u['model_calls']}, {','.join(u['tools']) or '-'}" if u else "-")
        print(f"| {t['turn']} | " + " | ".join(tok) + " | " + " | ".join(calls) + " |")

    print("\n| part | " + " | ".join(f"{n} lines" for n, _ in runs) + " |\n|---|" + "---|" * len(runs))
    for p in PARTS + ["total"]:
        print(f"| {p} | " + " | ".join(str(r["lines_of_code"].get(p, "-")) for _, r in runs) + " |")
    print("(SDK column = lines tagged `# replaces: <part>`; its file total is in the sdk.json lines_of_code)")

    for n, r in runs:
        print(f"\n{n} checks:")
        for c in r["checks"]:
            print(f"- [{'x' if c['passed'] else ' '}] {c['id']}: {c['detail']}")


if __name__ == "__main__":
    main()
