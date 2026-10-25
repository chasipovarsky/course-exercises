"""Caching on Gemini: there is no cache_control and we do not create explicit cache objects.

Implicit caching is on by default for Gemini 2.5+ (minimum 4,096 prompt tokens on
gemini-3.8-flash, per https://ai.google.dev/gemini-api/docs/caching). It only hits when the
request *prefix* is byte-identical, so this file's job is to keep the prefix frozen:
one GenerateContentConfig per agent (system instruction + sorted declarations), history
append-only. Hits are measured from usage_metadata.cached_content_token_count.
Explicit caching (CachedContent) is not used: no Google page confirms it on the free tier.
"""
from __future__ import annotations

from google.genai import types

from . import gemini_tools
from .gemini_tokens import THINKING_LEVEL
from .log import log

IMPLICIT_MIN_TOKENS = 4096


def frozen_config(system_prompt: str) -> types.GenerateContentConfig:
    log("gemini_caching.py", "built frozen config: system + tools prefix, implicit caching only")
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=gemini_tools.declarations(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        max_output_tokens=16000,
    )


def summary_config() -> types.GenerateContentConfig:
    """Compaction request: no tools, no system prompt (a different prefix, not expected to hit the cache)."""
    return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                                       max_output_tokens=16000)


def report(usage, label: str) -> None:
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    if cached:
        note = f"implicit cache hit {cached}/{prompt} prompt tokens"
    elif prompt < IMPLICIT_MIN_TOKENS:
        note = f"no hit: prompt {prompt} < {IMPLICIT_MIN_TOKENS} min for implicit caching"
    else:
        note = "no hit (implicit caching is best-effort)"
    log("gemini_caching.py", f"{label}: {note}")
