"""Gemini token usage, per model call and per user turn, from response.usage_metadata.

prompt_token_count already includes cached_content_token_count, so uncached input is the
difference. Thinking tokens (thoughts_token_count) are billed as output, so they count as output.
Free tier: real cost is 0. theoretical_cost_usd prices the same tokens at the paid Standard tier.
"""
from __future__ import annotations

from .log import log

MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "MEDIUM"  # closest match to effort="medium" on the Anthropic side

# USD per 1M tokens, gemini-3.8-flash, paid Standard tier, prices valid through 2026-12-31.
# Source: https://ai.google.dev/gemini-api/docs/pricing (page "Last updated 2026-10-19 UTC", read 2026-10-22)
PRICE = {"input": 0.75, "output": 3.75, "cached": 0.075}
PRICE_SOURCE = ("https://ai.google.dev/gemini-api/docs/pricing, page last updated 2026-10-19 UTC, read 2026-10-22; "
                "gemini-3.8-flash Standard tier, valid through 2026-12-31")


def theoretical_cost_usd(uncached: int, cached: int, output: int) -> float:
    return (uncached * PRICE["input"] + cached * PRICE["cached"] + output * PRICE["output"]) / 1e6


class GeminiTurnMeter:
    def __init__(self):
        self.uncached = self.cached = self.output = 0
        self.calls = 0
        self.last_context_tokens = 0  # size of the most recent prompt, used by compaction

    def add(self, usage, label: str = "call") -> None:
        prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        output = (int(getattr(usage, "candidates_token_count", 0) or 0)
                  + int(getattr(usage, "thoughts_token_count", 0) or 0))
        uncached = prompt - cached
        self.uncached += uncached
        self.cached += cached
        self.output += output
        self.calls += 1
        self.last_context_tokens = prompt + output
        log("gemini_tokens.py", f"{label} #{self.calls}: prompt={prompt} (cached={cached}) out={output} "
                                f"$0 (theoretical ${theoretical_cost_usd(uncached, cached, output):.5f})")

    def summary(self) -> dict:
        prompt = self.uncached + self.cached
        return {"input_tokens": self.uncached, "cache_read_tokens": self.cached, "cache_write_tokens": 0,
                "output_tokens": self.output,
                "cache_hit_pct": round(100 * self.cached / prompt, 1) if prompt else 0.0,
                "model_calls": self.calls, "cost_usd": 0.0,
                "theoretical_cost_usd": round(theoretical_cost_usd(self.uncached, self.cached, self.output), 6)}
