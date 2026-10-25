"""The shared tool schemas as Gemini function declarations, and dispatch of function_call parts."""
from __future__ import annotations

import json

from google.genai import types

from shared.toolkit import SPECS, Toolkit

from .log import log


def declarations() -> list[types.Tool]:
    # same JSON schemas as tools.py, sorted the same way so the prefix stays identical between calls
    decls = [types.FunctionDeclaration(name=s["name"], description=s["description"],
                                       parameters_json_schema=s["input_schema"])
             for s in sorted(SPECS, key=lambda s: s["name"])]
    return [types.Tool(function_declarations=decls)]


def execute(toolkit: Toolkit, call: dict) -> dict:
    """One function_call part -> one function_response part. Failures go back as {"error": ...}."""
    name, args = call["name"], call.get("args") or {}
    try:
        response = {"result": toolkit.run(name, args)}
    except (ValueError, TypeError) as e:  # bad arguments: the model can fix these itself
        response = {"error": str(e)}
    except Exception as e:  # network failure etc.: report it, keep the loop alive
        response = {"error": f"{type(e).__name__}: {e}"}
    text = response.get("result", response.get("error", ""))
    log("gemini_tools.py", f"{name}({json.dumps(args, ensure_ascii=False)[:120]}) -> "
                           f"{'ERROR ' if 'error' in response else ''}{len(text)} chars")
    fr = {"name": name, "response": response}
    if call.get("id"):
        fr["id"] = call["id"]
    return {"function_response": fr}
