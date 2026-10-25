import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "API"), str(ROOT)]

from harness import log  # noqa: E402

log.QUIET = True

HEBCAL_FIXTURE = {"items": [
    {"title": "Erev Pesach", "date": "2027-04-21", "category": "holiday"},
    {"title": "Pesach I", "date": "2027-04-22", "category": "holiday", "yomtov": True},
    {"title": "Pesach VII", "date": "2027-04-28", "category": "holiday", "yomtov": True},
]}


def fake_fetch(start, end):
    return HEBCAL_FIXTURE


def usage(inp=100, out=20, cr=0, cw=0):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=cr,
                           cache_creation_input_tokens=cw)


def resp(*blocks, stop="end_turn", **u):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage(**u))


def text(t):
    return {"type": "text", "text": t}


def tool_use(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp}


class FakeClient:
    """Stands in for anthropic.Anthropic(): returns scripted responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"
