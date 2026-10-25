"""Study-planner agent on the Claude Agent SDK: Claude Code runs the harness, we configure it."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock, ToolUseBlock

from shared.prompts import EFFORT, MODEL, PRICE, SYSTEM_PROMPT
from shared.toolkit import Toolkit
from tools import ALLOWED, make_server, write_claude_md


SESSIONS_FILE = "sdk_sessions.json"  # {our session name: Claude Code session_id}


def _cost(u: dict) -> float:
    return (u["input_tokens"] * PRICE["input"] + u["output_tokens"] * PRICE["output"]
            + u["cache_write_tokens"] * PRICE["cache_write"] + u["cache_read_tokens"] * PRICE["cache_read"]) / 1e6


def saved_session(data_dir: str | Path, name: str) -> str | None:
    """The Claude Code session_id last saved for this session name, to pass as resume=."""
    p = Path(data_dir) / SESSIONS_FILE
    return json.loads(p.read_text(encoding="utf-8")).get(name) if p.exists() else None


def _save_session(data_dir: Path, name: str, session_id: str) -> None:
    p = data_dir / SESSIONS_FILE
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data[name] = session_id
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")


class SdkAgent:
    def __init__(self, data_dir: str | Path, session_id: str, today: str, resume: str | None = None):
        cwd = Path(data_dir)
        self.cwd, self.name = cwd, session_id
        self.toolkit = Toolkit(cwd, clock=lambda: f"{today}T{datetime.now():%H:%M:%S}")
        write_claude_md(self.toolkit, cwd)  # replaces: memory.py (what is remembered gets loaded next session)
        self.options = ClaudeAgentOptions(
            model=MODEL,
            effort=EFFORT,
            system_prompt=SYSTEM_PROMPT + f"\nToday is {today}.",  # replaces: context.py
            tools=[],  # no built-in Claude Code tools: only ours
            mcp_servers={"planner": make_server(self.toolkit, cwd)},  # replaces: tools.py
            allowed_tools=ALLOWED,  # replaces: tools.py
            strict_mcp_config=True,
            permission_mode="dontAsk",
            setting_sources=["project"],  # replaces: memory.py (loads cwd/CLAUDE.md)
            cwd=str(cwd),
            resume=resume,  # replaces: history.py
        )
        self.client: ClaudeSDKClient | None = None
        self.session_id = resume
        self.cost_so_far = 0.0  # ResultMessage.total_cost_usd is cumulative per session

    async def turn(self, user_text: str) -> dict:
        t0 = time.perf_counter()
        if self.client is None:
            self.client = ClaudeSDKClient(self.options)  # replaces: compaction.py, caching.py (both automatic)
            await self.client.connect()
        await self.client.query(user_text)  # replaces: loop.py
        text, used, call_ids, result = [], [], set(), None
        async for msg in self.client.receive_response():  # replaces: loop.py
            if isinstance(msg, AssistantMessage):
                call_ids.add(msg.message_id)
                for b in msg.content:
                    if isinstance(b, ToolUseBlock):
                        used.append(b.name.removeprefix("mcp__planner__"))
                    elif isinstance(b, TextBlock):
                        text.append(b.text)
            elif isinstance(msg, ResultMessage):
                result = msg
        self.session_id = result.session_id  # replaces: history.py (resume=session_id later)
        _save_session(self.cwd, self.name, result.session_id)  # replaces: history.py (persisted for resume)
        u = result.usage or {}  # replaces: tokens.py
        usage = {"input_tokens": u.get("input_tokens", 0), "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                 "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
                 "output_tokens": u.get("output_tokens", 0)}
        total_in = usage["input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"]
        turn_cost = (result.total_cost_usd or 0.0) - self.cost_so_far  # replaces: tokens.py
        self.cost_so_far = result.total_cost_usd or 0.0
        return {"response": result.result or "\n".join(text), "tools": used, **usage,
                "cache_hit_pct": round(100 * usage["cache_read_tokens"] / total_in, 1) if total_in else 0.0,
                "model_calls": len(call_ids), "cost_usd": round(turn_cost, 5),
                "cost_usd_at_5m_cache_prices": round(_cost(usage), 5), "sdk_num_turns": result.num_turns,
                "session_id": result.session_id, "seconds": round(time.perf_counter() - t0, 2)}

    async def close(self) -> None:
        if self.client is not None:
            await self.client.disconnect()
