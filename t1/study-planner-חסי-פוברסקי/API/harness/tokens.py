"""Token usage and cost, per model call and per user turn."""
from __future__ import annotations

from shared.prompts import PRICE

from .log import log

FIELDS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")


def cost_usd(u: dict) -> float:
    return (u["input_tokens"] * PRICE["input"] + u["output_tokens"] * PRICE["output"]
            + u["cache_creation_input_tokens"] * PRICE["cache_write"]
            + u["cache_read_input_tokens"] * PRICE["cache_read"]) / 1e6


class TurnMeter:
    def __init__(self):
        self.usage = dict.fromkeys(FIELDS, 0)
        self.calls = 0
        self.last_context_tokens = 0  # size of the most recent prompt, used by compaction

    def add(self, usage, label: str = "call") -> None:
        u = {f: int(getattr(usage, f, 0) or 0) for f in FIELDS}
        for f in FIELDS:
            self.usage[f] += u[f]
        self.calls += 1
        self.last_context_tokens = (u["input_tokens"] + u["cache_read_input_tokens"]
                                    + u["cache_creation_input_tokens"] + u["output_tokens"])
        log("tokens.py", f"{label} #{self.calls}: in={u['input_tokens']} cache_read={u['cache_read_input_tokens']} "
                         f"cache_write={u['cache_creation_input_tokens']} out={u['output_tokens']} "
                         f"${cost_usd(u):.4f}")

    def summary(self) -> dict:
        u = self.usage
        total_in = u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"]
        return {"input_tokens": u["input_tokens"], "cache_read_tokens": u["cache_read_input_tokens"],
                "cache_write_tokens": u["cache_creation_input_tokens"], "output_tokens": u["output_tokens"],
                "cache_hit_pct": round(100 * u["cache_read_input_tokens"] / total_in, 1) if total_in else 0.0,
                "model_calls": self.calls, "cost_usd": round(cost_usd(u), 5)}
