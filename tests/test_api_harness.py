"""Offline tests for the API harness: a fake client, no key, no network."""
import json
import sys

from conftest import ROOT, FakeClient, fake_fetch, resp, text, tool_use

from harness import caching, compaction, context
from harness.history import History
from harness.tokens import TurnMeter, cost_usd

sys.path.insert(0, str(ROOT / "API"))
from agent import ApiAgent  # noqa: E402


def make_agent(client, data_dir, session="s1", threshold=30000):
    return ApiAgent(client, data_dir, session, "2027-04-12", threshold, fetch=fake_fetch)


def test_loop_runs_parallel_tools_and_returns_all_results_in_one_message(data_dir):
    client = FakeClient([
        resp(tool_use("a", "add_task", course="Algo", title="Final", kind="exam", due="2027-04-29", hours=12),
             tool_use("b", "add_task", course="OS", title="HW4", kind="assignment", due="2027-04-20", hours=5),
             stop="tool_use"),
        resp(text("Saved both."), stop="end_turn"),
    ])
    agent = make_agent(client, data_dir)
    r = agent.turn("add my two tasks")

    assert r["response"] == "Saved both."
    assert r["tools"] == ["add_task", "add_task"]
    assert r["model_calls"] == 2
    second = client.requests[1]["messages"]
    results = second[-1]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result"]  # one user message, both results
    assert {b["tool_use_id"] for b in results} == {"a", "b"}
    assert len(agent.toolkit.planner.tasks()) == 2


def test_tool_error_goes_back_to_model_as_is_error(data_dir):
    client = FakeClient([
        resp(tool_use("x", "update_task", task_id="T99", status="done"), stop="tool_use"),
        resp(text("No such task."), stop="end_turn"),
    ])
    make_agent(client, data_dir).turn("mark T99 done")
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "no task T99" in result["content"]


def test_history_is_append_only_and_resumable(data_dir):
    client = FakeClient([resp(text("hi"), stop="end_turn"), resp(text("again"), stop="end_turn")])
    agent = make_agent(client, data_dir)
    agent.turn("hello")
    path = agent.history.path
    first = path.read_text(encoding="utf-8")
    agent.turn("hello again")
    assert path.read_text(encoding="utf-8").startswith(first)  # only appended

    resumed = History(data_dir / "sessions", "s1")
    assert len(resumed.messages) == 4
    assert resumed.messages[-1]["content"][0]["text"] == "again"


def test_compaction_summarizes_past_threshold(data_dir):
    client = FakeClient([
        resp(text("one"), stop="end_turn", inp=500, out=100),
        resp(text("two"), stop="end_turn", inp=500, out=100),
        resp(text("<summary>user has an exam T1 on 2027-04-29</summary>"), stop="end_turn"),
        resp(text("three"), stop="end_turn"),
    ])
    agent = make_agent(client, data_dir, threshold=550)
    agent.turn("first")
    agent.turn("second")  # previous context 600 >= 550, but only 2 messages: not yet
    agent.turn("third")   # now compacts first
    assert agent.compactions == 1
    summary_req = client.requests[2]
    assert "tools" not in summary_req and "system" not in summary_req
    assert "<transcript>" in summary_req["messages"][0]["content"][0]["text"]
    assert "conversation_summary" in client.requests[3]["messages"][0]["content"][0]["text"]
    records = [json.loads(line) for line in agent.history.path.read_text(encoding="utf-8").splitlines()]
    assert any(r["type"] == "compaction" for r in records)
    assert sum(r["type"] == "message" for r in records) == 6  # nothing deleted from disk


def test_cache_breakpoints_on_request_but_not_in_history(data_dir):
    client = FakeClient([resp(text("ok"), stop="end_turn")])
    agent = make_agent(client, data_dir)
    agent.turn("hi")
    req = client.requests[0]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in json.dumps(agent.history.messages)
    assert [t["name"] for t in req["tools"]] == sorted(t["name"] for t in req["tools"])  # stable prefix


def test_cache_marker_skips_thinking_blocks():
    msgs = [{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "thinking", "thinking": ""}]}]
    marked = caching.mark_messages(msgs)
    assert "cache_control" in marked[0]["content"][0]
    assert "cache_control" not in marked[0]["content"][1]


def test_context_has_prefs_and_short_digest_not_full_task_list(data_dir):
    agent = make_agent(FakeClient([]), data_dir)
    tk = agent.toolkit
    tk.run("set_preference", {"key": "daily_study_hours", "value": "3"})
    for i in range(5):
        tk.run("add_task", {"course": f"C{i}", "title": "x", "kind": "exam", "due": f"2027-05-0{i + 1}", "hours": 2})
    msg = context.build_user_message(tk, "plan", "2027-04-12")
    block = msg["content"][0]["text"]
    assert "today: 2027-04-12" in block and "pref daily_study_hours=3" in block
    assert block.count("due ") == 3 and "+2 more open tasks" in block
    assert msg["content"][1]["text"] == "plan"


def test_memory_contradiction_keeps_old_value(data_dir):
    agent = make_agent(FakeClient([]), data_dir)
    tk = agent.toolkit
    assert tk.run("set_preference", {"key": "daily_study_hours", "value": "3 hours"}).startswith("saved")
    assert "3.0 -> 2.0" in tk.run("set_preference", {"key": "daily_study_hours", "value": "2"})
    assert tk.memory.preferences()["daily_study_hours"] == 2
    assert tk.memory.history()[0]["value"] == 3
    assert "old daily_study_hours=3" in tk.run("recall", {"query": "hours"})


def test_tokens_cost_and_cache_pct():
    m = TurnMeter()
    from conftest import usage
    m.add(usage(inp=1000, out=100, cr=9000, cw=0))
    s = m.summary()
    assert s["cache_hit_pct"] == 90.0
    assert abs(s["cost_usd"] - cost_usd({"input_tokens": 1000, "output_tokens": 100,
                                          "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 0})) < 1e-9
    assert abs(s["cost_usd"] - (1000 * 4 + 100 * 20 + 9000 * 0.2) / 1e6) < 1e-9


def test_render_transcript_skips_thinking():
    msgs = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "secret"}, text("hi")]}]
    assert "secret" not in compaction.render(msgs)
