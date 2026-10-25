"""Study-planner agent on the raw Messages API: wires the harness pieces together."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from harness import compaction, context
from harness.history import History
from harness.log import log
from harness.loop import ModelCaller, run_turn
from harness.tokens import TurnMeter
from shared.prompts import SYSTEM_PROMPT
from shared.toolkit import Toolkit


class ApiAgent:
    def __init__(self, client, data_dir: str | Path, session_id: str, today: str,
                 compaction_threshold: int = 30000, fetch=None):
        self.client = client
        self.today = today
        self.threshold = compaction_threshold
        self.toolkit = Toolkit(data_dir, clock=lambda: f"{today}T{datetime.now():%H:%M:%S}", fetch=fetch)
        self.history = History(Path(data_dir) / "sessions", session_id)
        self.last_meter = TurnMeter()
        self.compactions = 0
        log("agent.py", f"session {session_id} ready (today={today}, data={data_dir})")

    def turn(self, user_text: str) -> dict:
        t0 = time.perf_counter()
        meter = TurnMeter()
        call = ModelCaller(self.client, meter, SYSTEM_PROMPT)
        meter.last_context_tokens = self.last_meter.last_context_tokens
        if compaction.maybe_compact(call, self.history, meter, self.threshold):
            self.compactions += 1
        msg = context.build_user_message(self.toolkit, user_text, self.today)
        out = run_turn(call, self.history, self.toolkit, msg)
        self.last_meter = meter
        return {"response": out["text"], "tools": out["tools"], **meter.summary(),
                "seconds": round(time.perf_counter() - t0, 2)}
