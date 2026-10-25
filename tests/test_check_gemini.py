"""check_gemini.py diagnostics, offline: fake client, no key, no network."""
import importlib.util
from types import SimpleNamespace

import pytest
from conftest import ROOT

pytest.importorskip("google.genai")
from google.genai import errors, types  # noqa: E402

spec = importlib.util.spec_from_file_location("check_gemini", ROOT / "check_gemini.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

QUOTA_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
         "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
         "quotaDimensions": {"model": "gemini-3.8-flash", "location": "global"}, "quotaValue": "0"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}}


class Fake:
    def __init__(self, listing, answers):
        self.listing, self.answers = listing, answers
        self.models = self

    def list(self):
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing

    def generate_content(self, model, contents, config):
        r = self.answers[model]
        if isinstance(r, Exception):
            raise r
        return r


def ok_resp(text="Hi!"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=2))


def test_reports_ok_quota_zero_and_missing_model():
    listing = [SimpleNamespace(name="models/gemini-3.5-flash-lite", supported_actions=["generateContent"]),
               SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"])]
    fake = Fake(listing, {
        "gemini-3.5-flash-lite": ok_resp(),
        "gemini-3.8-flash": errors.ClientError(429, QUOTA_429),
        "gemini-9-nope": errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND", "message": "no"}}),
    })
    results = check.run(fake, ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-9-nope"])
    out = "\n".join("\n".join(lines) for _, lines in results)
    assert results[0][0] and "2 models visible" in out and "gemini-9-nope: not listed" in out
    assert results[1][0] and "gemini-3.5-flash-lite: OK" in out
    assert not results[2][0] and "free quota for this model is 0" in out
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in out and "retryDelay=41s" in out
    assert not results[3][0] and "model not found" in out


def test_network_block_is_named_as_such():
    fake = Fake(ConnectionError("[WinError 10061] connection refused"), {"m": OSError("getaddrinfo failed")})
    results = check.run(fake, ["m"])
    assert all(not ok for ok, _ in results)
    assert all("BLOCKED: network" in lines[0] for _, lines in results)


def test_key_never_printed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-value-for-test")
    err = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                             "message": "API key not valid: fake-key-value-for-test",
                                             "details": [{"reason": "API_KEY_INVALID"}]}})
    ok, lines = check.check_hi(Fake([], {"m": err}), "m")
    text = "\n".join(lines)
    assert not ok and "key rejected" in text
    assert "fake-key-value-for-test" not in text
