"""Caching on Gemini: there is no cache_control and we do not create explicit cache objects.

Implicit caching is on by default for Gemini 2.5+ (per https://ai.google.dev/gemini-api/docs/caching;
minimum prompt size depends on the model). It only hits when the request *prefix* is byte-identical,
so this file's job is to keep the prefix frozen: one GenerateContentConfig per agent (system
instruction + sorted declarations), history append-only. Hits are measured from
usage_metadata.cached_content_token_count.
Explicit caching (CachedContent) is not used: no Google page confirms it on the free tier.
"""
from __future__ import annotations

from google.genai import types

from . import gemini_tools
from .gemini_tokens import THINKING_LEVEL
from .log import log

# "Min token limit" table on the caching page (read 2026-10-22); models not listed there are unknown
IMPLICIT_MIN_TOKENS = {"gemini-3.8-flash": 4096, "gemini-3.6-flash": 4096,
                       "gemini-2.5-flash": 2048, "gemini-2.5-pro": 2048}


def thinking_config(model: str) -> types.ThinkingConfig | None:
    """thinking_level exists on Gemini 3+; older models keep their default thinking."""
    if model.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    return None


def frozen_config(system_prompt: str, model: str) -> types.GenerateContentConfig:
    log("gemini_caching.py", f"built frozen config for {model}: system + tools prefix, implicit caching only")
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=gemini_tools.declarations(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=thinking_config(model),
        max_output_tokens=16000,
    )


def summary_config(model: str) -> types.GenerateContentConfig:
    """Compaction request: no tools, no system prompt (a different prefix, not expected to hit the cache)."""
    return types.GenerateContentConfig(thinking_config=thinking_config(model), max_output_tokens=16000)


def report(usage, label: str, model: str) -> None:
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    minimum = IMPLICIT_MIN_TOKENS.get(model)
    if cached:
        note = f"implicit cache hit {cached}/{prompt} prompt tokens"
    elif minimum and prompt < minimum:
        note = f"no hit: prompt {prompt} < {minimum} min for implicit caching"
    else:
        note = "no hit (implicit caching is best-effort)"
    log("gemini_caching.py", f"{label}: {note}")
