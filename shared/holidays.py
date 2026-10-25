"""External tool backend: Israeli holidays from the public Hebcal REST API (no key).

Responses are cached on disk per date range, so a re-plan of the same week never
hits the network twice. `fetch` is injectable so offline tests never touch the web.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date

from .store import Store

HEBCAL_URL = "https://www.hebcal.com/hebcal"


def _http_fetch(start: str, end: str) -> dict:
    params = {"v": 1, "cfg": "json", "maj": "on", "min": "off", "mod": "off", "nx": "off",
              "ss": "off", "mf": "off", "c": "off", "i": "on", "start": start, "end": end}
    url = HEBCAL_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


class Holidays:
    def __init__(self, store: Store, fetch=None):
        self.store = store
        self.fetch = fetch or _http_fetch
        self.network_calls = 0

    def between(self, start: str, end: str) -> list[dict]:
        """[{date, title, day_off}] for start..end inclusive. day_off = yom tov (no study)."""
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        if e < s:
            raise ValueError("end is before start")
        if (e - s).days > 62:
            raise ValueError("range too long; ask for at most ~2 months")
        cache = f"cache/hebcal_{start}_{end}.json"
        raw = self.store.load(cache, None)
        if raw is None:
            raw = self.fetch(start, end)
            self.network_calls += 1
            self.store.save(cache, raw)
        out = []
        for it in raw.get("items", []):
            d = it.get("date", "")[:10]
            if start <= d <= end and it.get("category") == "holiday":
                out.append({"date": d, "title": it.get("title", ""), "day_off": bool(it.get("yomtov"))})
        return out
