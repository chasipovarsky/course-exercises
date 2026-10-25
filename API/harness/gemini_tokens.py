"""Gemini token usage, per model call and per user turn, from response.usage_metadata.

prompt_token_count already includes cached_content_token_count, so uncached input is the
difference. Thinking tokens (thoughts_token_count) are billed as output, so they count as output.
Free tier: real cost is 0. theoretical_cost_usd prices the same tokens at the paid Standard tier;
for a model with no known price it is None rather than a guess.
"""
from __future__ import annotations

from .log import log

DEFAULT_MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "MEDIUM"  # closest match to effort="medium" on the Anthropic side

# USD per 1M tokens, paid Standard tier, text input, prompts <= 200k tokens.
# Source: https://ai.google.dev/gemini-api/docs/pricing (page "Last updated 2026-10-19 UTC", read 2026-10-22).
# 3.8 / 3.6 Flash prices are valid through 2026-12-31 (they double on 2027-01-01).
PRICES = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cached": 0.075},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75, "cached": 0.075},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached": 0.025},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached": 0.125},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached": 0.01},
}
PRICE_SOURCE = ("https://ai.google.dev/gemini-api/docs/pricing, page last updated 2026-10-19 UTC, read 2026-10-22; "
                "Standard tier, text, prompts <= 200k; 3.8/3.6 Flash prices valid through 2026-12-31")


def price_for(model: str) -> dict | None:
    return PRICES.get(model)


def theoretical_cost_usd(uncached: int, cached: int, output: int, price: dict | None) -> float | None:
    if price is None:
        return None
    return (uncached * price["input"] + cached * price["cached"] + output * price["output"]) / 1e6


class GeminiTurnMeter:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.price = price_for(model)
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
        cost = theoretical_cost_usd(uncached, cached, output, self.price)
        log("gemini_tokens.py", f"{label} #{self.calls}: prompt={prompt} (cached={cached}) out={output} "
                                f"$0 (theoretical {'$%.5f' % cost if cost is not None else 'n/a: no price'})")

    def summary(self) -> dict:
        prompt = self.uncached + self.cached
        cost = theoretical_cost_usd(self.uncached, self.cached, self.output, self.price)
        return {"input_tokens": self.uncached, "cache_read_tokens": self.cached, "cache_write_tokens": 0,
                "output_tokens": self.output,
                "cache_hit_pct": round(100 * self.cached / prompt, 1) if prompt else 0.0,
                "model_calls": self.calls, "cost_usd": 0.0,
                "theoretical_cost_usd": round(cost, 6) if cost is not None else None}
