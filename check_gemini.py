"""Diagnose Gemini access: what works, what is blocked, and why.

  ..\\.venv\\Scripts\\python check_gemini.py
  ..\\.venv\\Scripts\\python check_gemini.py gemini-3.8-flash gemini-3.5-flash-lite gemini-3.1-flash-lite

1. models.list: is the API reachable with this key, and which candidate models does it list?
2. a one-word "hi" request to each candidate model (default: three stable free-tier models).
For 429s it prints the quota that ran out (quotaId / quotaValue), so you can see whether the
limit is per minute, per day, or 0 (no free quota for that model). Key: GEMINI_API_KEY only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent / "API"), str(Path(__file__).resolve().parent)]

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite"]
NETWORK_HINTS = ("connect", "timeout", "ssl", "proxy", "getaddrinfo", "name resolution", "network", "refused",
                 "certificate", "unreachable")


def quota_violations(e) -> list[str]:
    """QuotaFailure.violations from a 429 body, one line each."""
    body = getattr(e, "details", None) or {}
    details = body.get("error", {}).get("details", []) if isinstance(body, dict) else []
    out = []
    for d in details:
        if "QuotaFailure" in d.get("@type", ""):
            for v in d.get("violations", []):
                dims = ",".join(f"{k}={val}" for k, val in (v.get("quotaDimensions") or {}).items())
                out.append(f"quotaId={v.get('quotaId')} limit={v.get('quotaValue')} metric={v.get('quotaMetric')}"
                           + (f" [{dims}]" if dims else ""))
        elif "RetryInfo" in d.get("@type", ""):
            out.append(f"retryDelay={d.get('retryDelay')}")
    return out


def classify(e: Exception) -> tuple[str, list[str]]:
    """(verdict, extra lines). Never includes the key."""
    from google.genai import errors
    from harness.gemini_loop import redact

    if isinstance(e, errors.APIError):
        status = getattr(e, "status", "") or ""
        msg = redact(str(getattr(e, "message", "") or e))[:300]
        if e.code == 429:
            v = quota_violations(e)
            zero = any("limit=0" in line for line in v)
            verdict = "BLOCKED: quota exhausted (429)" + (" - free quota for this model is 0" if zero else "")
            return verdict, v or [msg]
        if e.code == 400 and "API_KEY" in (msg + str(getattr(e, "details", ""))):
            return "BLOCKED: key rejected (400 API_KEY_INVALID)", [msg]
        if e.code in (401, 403):
            return f"BLOCKED: permission denied ({e.code} {status})", [msg]
        if e.code == 404:
            return "UNAVAILABLE: model not found for this key (404)", [msg]
        return f"FAILED: HTTP {e.code} {status}", [msg]
    text = f"{type(e).__name__}: {e}"
    if isinstance(e, OSError) or any(h in text.lower() for h in NETWORK_HINTS):
        return "BLOCKED: network (no connection to generativelanguage.googleapis.com)", [redact(text)[:300]]
    return f"FAILED: {type(e).__name__}", [redact(text)[:300]]


def check_list(client, candidates: list[str]) -> tuple[bool, list[str]]:
    try:
        models = list(client.models.list())
    except Exception as e:
        verdict, extra = classify(e)
        return False, [f"models.list: {verdict}", *("    " + x for x in extra)]
    names = {m.name.removeprefix("models/"): m for m in models}
    lines = [f"models.list: OK, {len(names)} models visible to this key"]
    for c in candidates:
        m = names.get(c)
        if m is None:
            lines.append(f"    {c}: not listed")
        else:
            actions = getattr(m, "supported_actions", None) or []
            lines.append(f"    {c}: listed" + (", generateContent" if "generateContent" in actions else ""))
    return True, lines


def check_hi(client, model: str) -> tuple[bool, list[str]]:
    from google.genai import types
    try:
        r = client.models.generate_content(model=model, contents="hi",
                                           config=types.GenerateContentConfig(max_output_tokens=256))
    except Exception as e:
        verdict, extra = classify(e)
        return False, [f"{model}: {verdict}", *("    " + x for x in extra)]
    text = (r.text or "").strip().replace("\n", " ")[:60] if r.candidates else ""
    u = r.usage_metadata
    return True, [f"{model}: OK, replied {text!r} (prompt={getattr(u, 'prompt_token_count', '?')} tokens)"]


def run(client, models: list[str]) -> list[tuple[bool, list[str]]]:
    return [check_list(client, models)] + [check_hi(client, m) for m in models]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY is not set. In cmd: set GEMINI_API_KEY=...")
        return 2
    from google import genai
    client = genai.Client(api_key=key)
    models = sys.argv[1:] or DEFAULT_MODELS
    results = run(client, models)
    for _, lines in results:
        print("\n".join(lines))
    ok = [r[1][0].split(":")[0] for r in results[1:] if r[0]]
    print(f"\nsummary: models.list {'OK' if results[0][0] else 'BLOCKED'}; "
          f"{len(ok)}/{len(models)} models answered" + (f" ({', '.join(ok)})" if ok else ""))
    if ok:
        print(f"run the scenario with: ..\\.venv\\Scripts\\python API\\main.py --provider gemini --model {ok[0]} "
              f"--scenario scenario.json --out results\\api_gemini.json")
    return 0 if all(r[0] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
