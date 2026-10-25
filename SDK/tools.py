"""The shared toolkit exposed to Claude Code as an in-process MCP server."""
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from shared.toolkit import SPECS, Toolkit, fmt_value

SERVER = "planner"
MEMORY_TOOLS = {"set_preference", "remember"}


def write_claude_md(toolkit: Toolkit, cwd: Path) -> None:
    """Long-term memory as CLAUDE.md; Claude Code loads it at session start (setting_sources=["project"])."""
    mem = toolkit.memory
    lines = ["# Study planner memory (auto-generated, do not edit)", "", "## Current preferences"]
    lines += [f"- {k}: {fmt_value(v)}" for k, v in mem.preferences().items()] or ["- none yet"]
    if mem.notes():
        lines += ["", "## Notes"] + [f"- [{n['topic']}] {n['text']}" for n in mem.notes()[-10:]]
    if mem.history():
        lines += ["", f"({len(mem.history())} superseded values: use recall to see them)"]
    (cwd / "CLAUDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_server(toolkit: Toolkit, cwd: Path):  # replaces: tools.py
    def wrap(spec):
        @tool(spec["name"], spec["description"], spec["input_schema"])
        async def handler(args):
            try:
                text = toolkit.run(spec["name"], args)
            except Exception as e:  # replaces: tools.py (errors go back to the model as is_error)
                return {"content": [{"type": "text", "text": f"error: {e}"}], "is_error": True}
            if spec["name"] in MEMORY_TOOLS:
                write_claude_md(toolkit, cwd)  # replaces: memory.py
            return {"content": [{"type": "text", "text": text}]}
        return handler

    return create_sdk_mcp_server(name=SERVER, version="1.0.0", tools=[wrap(s) for s in SPECS])


ALLOWED = [f"mcp__{SERVER}__{s['name']}" for s in SPECS]
