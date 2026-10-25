"""SDK version: configuration and tool wrappers, checked without starting Claude Code."""
import asyncio
import importlib.util

import pytest
from conftest import ROOT, fake_fetch

pytest.importorskip("claude_agent_sdk")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "SDK"))
    import sys
    sys.modules.pop("tools", None)  # make sure SDK/tools.py, not some other "tools"
    return load("sdk_tools", ROOT / "SDK" / "tools.py"), load("sdk_agent", ROOT / "SDK" / "agent.py")


def test_options_are_restricted(sdk, data_dir):
    _, agent_mod = sdk
    a = agent_mod.SdkAgent(data_dir, "s1", "2027-04-12")
    o = a.options
    assert o.strict_mcp_config is True
    assert o.tools == []
    assert o.setting_sources == ["project"]
    assert o.model == "claude-opus-5-5" and o.effort == "medium"
    assert all(t.startswith("mcp__planner__") for t in o.allowed_tools) and len(o.allowed_tools) == 8
    assert "Today is 2027-04-12" in o.system_prompt
    assert (data_dir / "CLAUDE.md").exists()


def test_tool_wrapper_reports_errors_and_writes_claude_md(sdk, data_dir):
    tools_mod, _ = sdk
    from shared.toolkit import Toolkit
    tk = Toolkit(data_dir, clock=lambda: "2027-04-12T10:00:00", fetch=fake_fetch)
    server = tools_mod.make_server(tk, data_dir)
    handlers = {t.name: t.handler for t in server_tools(tools_mod, tk, data_dir)}
    bad = asyncio.run(handlers["update_task"]({"task_id": "T9", "status": "done"}))
    assert bad["is_error"] is True
    ok = asyncio.run(handlers["set_preference"]({"key": "off_days", "value": "Fri,Sat"}))
    assert "saved" in ok["content"][0]["text"]
    assert "off_days: Fri,Sat" in (data_dir / "CLAUDE.md").read_text(encoding="utf-8")
    assert server is not None


def server_tools(tools_mod, tk, data_dir):
    """Rebuild the decorated tools the same way make_server does, to call their handlers directly."""
    captured = []
    real = tools_mod.create_sdk_mcp_server
    tools_mod.create_sdk_mcp_server = lambda name, version, tools: captured.extend(tools) or "srv"
    try:
        tools_mod.make_server(tk, data_dir)
    finally:
        tools_mod.create_sdk_mcp_server = real
    return captured


def test_every_harness_part_is_marked(sdk):
    import re
    text = (ROOT / "SDK" / "agent.py").read_text(encoding="utf-8") + (ROOT / "SDK" / "tools.py").read_text(encoding="utf-8")
    parts = set(re.findall(r"# replaces: .*", text))
    joined = " ".join(parts)
    for p in ["loop", "tools", "history", "memory", "context", "compaction", "caching", "tokens"]:
        assert f"{p}.py" in joined, p
