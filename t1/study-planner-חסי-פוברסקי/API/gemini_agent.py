"""Study-planner agent on the Gemini API: same harness layout, Gemini-specific loop/tools/caching/tokens.

history.py, memory.py, context.py and compaction.py are the same files the Anthropic agent uses.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from harness import compaction, context, gemini_caching
from harness.gemini_loop import GeminiCaller, run_turn
from harness.gemini_tokens import DEFAULT_MODEL, GeminiTurnMeter
from harness.history import History
from harness.log import log
from shared.prompts import SYSTEM_PROMPT
from shared.toolkit import Toolkit


class GeminiAgent:
    def __init__(self, client, data_dir: str | Path, session_id: str, today: str,
                 compaction_threshold: int = 30000, fetch=None, sleep=time.sleep,
                 model: str = DEFAULT_MODEL):
        self.client = client
        self.today = today
        self.threshold = compaction_threshold
        self.sleep = sleep
        self.model = model
        self.toolkit = Toolkit(data_dir, clock=lambda: f"{today}T{datetime.now():%H:%M:%S}", fetch=fetch)
        self.history = History(Path(data_dir) / "sessions", session_id, provider="gemini")
        self.config = gemini_caching.frozen_config(SYSTEM_PROMPT, model)  # built once: identical prefix every call
        self.last_meter = GeminiTurnMeter(model)
        self.compactions = 0
        log("gemini_agent.py", f"session {session_id} ready (model={model}, today={today}, data={data_dir})")

    def turn(self, user_text: str) -> dict:
        t0 = time.perf_counter()
        meter = GeminiTurnMeter(self.model)
        call = GeminiCaller(self.client, meter, self.config, sleep=self.sleep, model=self.model)
        meter.last_context_tokens = self.last_meter.last_context_tokens
        if compaction.maybe_compact(call, self.history, meter, self.threshold):
            self.compactions += 1
        msg = context.build_gemini_user_message(self.toolkit, user_text, self.today)
        out = run_turn(call, self.history, self.toolkit, msg)
        self.last_meter = meter
        return {"response": out["text"], "tools": out["tools"], **meter.summary(),
                "seconds": round(time.perf_counter() - t0, 2)}
