"""Offline tests for the Gemini provider: fake client returning real google.genai types, no key, no network."""
import asyncio
import json
import sys

import pytest
from conftest import ROOT, fake_fetch

pytest.importorskip("google.genai")
from google.genai import errors, types  # noqa: E402

from harness import gemini_caching, gemini_tools  # noqa: E402
from harness.gemini_tokens import PRICES, GeminiTurnMeter, theoretical_cost_usd  # noqa: E402

sys.path.insert(0, str(ROOT / "API"))
from gemini_agent import GeminiAgent  # noqa: E402

SIG = b"\x00\x01signature\xff"


def g_usage(prompt=100, cached=0, out=20, thoughts=5):
    return types.GenerateContentResponseUsageMetadata(prompt_token_count=prompt, cached_content_token_count=cached,
                                                      candidates_token_count=out, thoughts_token_count=thoughts)


def g_resp(*parts, **u):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason="STOP")],
        usage_metadata=g_usage(**u))


def g_text(t):
    return types.Part(text=t)


def g_call(name, id_=None, sig=None, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=id_), thought_signature=sig)


class FakeGemini:
    """Stands in for genai.Client(): client.models.generate_content(model, contents, config)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_agent(client, data_dir, session="s1", threshold=30000, sleeps=None):
    return GeminiAgent(client, data_dir, session, "2027-04-12", threshold, fetch=fake_fetch,
                       sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_loop_runs_parallel_calls_and_returns_all_responses_in_one_turn(data_dir):
    client = FakeGemini([
        g_resp(g_call("add_task", "c1", course="Algo", title="Final", kind="exam", due="2027-04-29", hours=12),
               g_call("add_task", "c2", course="OS", title="HW4", kind="assignment", due="2027-04-20", hours=5)),
        g_resp(g_text("Saved both.")),
    ])
    agent = make_agent(client, data_dir)
    r = agent.turn("add my two tasks")
    assert r["response"] == "Saved both."
    assert r["tools"] == ["add_task", "add_task"]
    assert r["model_calls"] == 2
    last = client.requests[1]["contents"][-1]
    assert last.role == "user"
    assert [p.function_response.id for p in last.parts] == ["c1", "c2"]  # one turn, both responses
    assert all("result" in p.function_response.response for p in last.parts)
    assert client.requests[0]["model"] == "gemini-3.8-flash"
    assert len(agent.toolkit.planner.tasks()) == 2


def test_tool_error_goes_back_as_error_field(data_dir):
    client = FakeGemini([g_resp(g_call("update_task", task_id="T99", status="done")), g_resp(g_text("No such task."))])
    make_agent(client, data_dir).turn("mark T99 done")
    fr = client.requests[1]["contents"][-1].parts[0].function_response
    assert "no task T99" in fr.response["error"]
    assert "result" not in fr.response


def test_429_is_retried_with_backoff(data_dir):
    rate = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    client = FakeGemini([rate, rate, g_resp(g_text("ok"))])
    sleeps = []
    r = make_agent(client, data_dir, sleeps=sleeps).turn("hi")
    assert r["response"] == "ok"
    assert sleeps == [5.0, 10.0]  # exponential backoff
    assert r["model_calls"] == 1  # retries are not extra model calls in the metrics


def test_429_uses_server_retry_delay(data_dir):
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}
    client = FakeGemini([errors.ClientError(429, body), g_resp(g_text("ok"))])
    sleeps = []
    make_agent(client, data_dir, sleeps=sleeps).turn("hi")
    assert sleeps == [18.0]


def test_other_client_errors_are_not_retried(data_dir):
    client = FakeGemini([errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID"}})])
    with pytest.raises(errors.ClientError):
        make_agent(client, data_dir).turn("hi")


def test_thought_signature_survives_history_on_disk(data_dir):
    client = FakeGemini([g_resp(g_call("list_tasks", "c1", sig=SIG)), g_resp(g_text("none"))])
    agent = make_agent(client, data_dir)
    agent.turn("what is open?")
    sent_back = client.requests[1]["contents"][1].parts[0]
    assert sent_back.thought_signature == SIG  # byte-identical after a dict round trip
    from harness.history import History
    resumed = History(data_dir / "sessions", "s1", provider="gemini")
    assert types.Content.model_validate(resumed.messages[1]).parts[0].thought_signature == SIG


def test_config_is_frozen_and_has_no_cache_control(data_dir):
    client = FakeGemini([g_resp(g_text("a")), g_resp(g_text("b"))])
    agent = make_agent(client, data_dir)
    agent.turn("one")
    agent.turn("two")
    c1, c2 = client.requests[0]["config"], client.requests[1]["config"]
    assert c1 is c2  # same object: identical prefix for implicit caching
    names = [d.name for d in c1.tools[0].function_declarations]
    assert names == sorted(names) and len(names) == 8
    assert c1.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert c1.automatic_function_calling.disable is True
    assert "cache_control" not in json.dumps(c1.model_dump(mode="json", exclude_none=True))


def test_tokens_split_cached_and_theoretical_cost():
    m = GeminiTurnMeter()
    m.add(g_usage(prompt=10000, cached=8000, out=300, thoughts=200))
    s = m.summary()
    assert (s["input_tokens"], s["cache_read_tokens"], s["output_tokens"]) == (2000, 8000, 500)
    assert s["cost_usd"] == 0.0 and s["cache_hit_pct"] == 80.0
    PRICE = PRICES["gemini-3.8-flash"]
    expected = (2000 * PRICE["input"] + 8000 * PRICE["cached"] + 500 * PRICE["output"]) / 1e6
    assert abs(s["theoretical_cost_usd"] - round(expected, 6)) < 1e-9
    assert abs(theoretical_cost_usd(2000, 8000, 500, PRICE) - expected) < 1e-12


def test_cache_report_explains_misses(capsys):
    from harness import log
    log.QUIET = False
    try:
        gemini_caching.report(g_usage(prompt=1000), "step 1", "gemini-3.8-flash")
        gemini_caching.report(g_usage(prompt=6000, cached=4096), "step 2", "gemini-3.8-flash")
    finally:
        log.QUIET = True
    err = capsys.readouterr().err
    assert "1000 < 4096" in err and "implicit cache hit 4096/6000" in err


def test_compaction_works_with_gemini_history(data_dir):
    client = FakeGemini([
        g_resp(g_text("one"), prompt=500, out=100, thoughts=0),
        g_resp(g_text("two"), prompt=500, out=100, thoughts=0),
        g_resp(g_text("<summary>exam T1 on 2027-04-29</summary>")),
        g_resp(g_text("three")),
    ])
    agent = make_agent(client, data_dir, threshold=550)
    agent.turn("first")
    agent.turn("second")
    agent.turn("third")
    assert agent.compactions == 1
    summary_req = client.requests[2]
    assert summary_req["config"].tools is None and summary_req["config"].system_instruction is None
    assert "<transcript>" in summary_req["contents"][0].parts[0].text
    assert "conversation_summary" in client.requests[3]["contents"][0].parts[0].text


def test_empty_candidate_ends_turn_without_appending(data_dir):
    blocked = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="SAFETY")],
                                            usage_metadata=g_usage())
    agent = make_agent(FakeGemini([blocked]), data_dir)
    r = agent.turn("hi")
    assert "SAFETY" in r["response"]
    assert len(agent.history.messages) == 1  # only the user message


def test_declarations_cover_every_shared_tool():
    from shared.toolkit import SPECS
    decls = gemini_tools.declarations()[0].function_declarations
    assert {d.name for d in decls} == {s["name"] for s in SPECS}
    assert all(d.parameters_json_schema for d in decls)


def test_key_only_from_gemini_env_and_never_in_results(tmp_path, data_dir, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("api_main", ROOT / "API" / "main.py")
    api_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_main)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "should-not-be-used")
    with pytest.raises(SystemExit):
        api_main.gemini_client()

    secret = "fake-gemini-key-for-offline-test"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    responses = [g_resp(g_text("ok")) for _ in range(7)]
    client = FakeGemini(responses)
    make = lambda sid, today, th: make_agent(client, data_dir, session=sid)  # noqa: E731
    out = tmp_path / "api_gemini.json"
    from shared.runner import run_scenario
    asyncio.run(run_scenario(make, str(ROOT / "scenario.json"), str(out), data_dir, "api", api_main.loc("gemini"),
                             {"provider": "gemini", "model": "gemini-3.8-flash"}))
    result = json.loads(out.read_text(encoding="utf-8"))
    assert secret not in out.read_text(encoding="utf-8")
    assert all(secret not in p.read_text(encoding="utf-8") for p in data_dir.rglob("*") if p.is_file())
    assert result["model"] == "gemini-3.8-flash" and "theoretical_cost_usd" in result["totals"]
    assert set(result) >= {"turns", "totals", "checks", "checks_passed", "lines_of_code"}  # same format as sdk.json
    assert result["lines_of_code"]["loop"] > 0


def test_429_body_is_logged_with_key_redacted(data_dir, capsys, monkeypatch):
    from harness import log
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-in-env-for-test")
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "message": "Quota exceeded for key fake-key-in-env-for-test",
                      "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                   "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                                   "quotaValue": "0"}]}]}}
    client = FakeGemini([errors.ClientError(429, body), g_resp(g_text("ok"))])
    log.QUIET = False
    try:
        make_agent(client, data_dir).turn("hi")
    finally:
        log.QUIET = True
    err = capsys.readouterr().err
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in err and '"quotaValue": "0"' in err
    assert "fake-key-in-env-for-test" not in err and "[REDACTED]" in err


def test_model_parameter_reaches_the_request(data_dir):
    client = FakeGemini([g_resp(g_text("ok"))])
    agent = GeminiAgent(client, data_dir, "s1", "2027-04-12", fetch=fake_fetch, sleep=lambda s: None,
                        model="gemini-3.5-flash-lite")
    r = agent.turn("hi")
    assert client.requests[0]["model"] == "gemini-3.5-flash-lite"
    assert client.requests[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert r["theoretical_cost_usd"] is not None
    other = GeminiAgent(FakeGemini([g_resp(g_text("ok"))]), data_dir / "x", "s1", "2027-04-12",
                        fetch=fake_fetch, model="gemini-unknown-model")
    assert other.config.thinking_config is None  # not a Gemini 3 model: keep the model's default thinking
    assert other.turn("hi")["theoretical_cost_usd"] is None  # no known price: not guessed
