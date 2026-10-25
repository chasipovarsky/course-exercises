"""Disk-backed state shared by both harnesses: tasks, plans, memory, external cache.

Everything lives under one data_dir so a run can start from a clean state by deleting it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class Store:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "cache").mkdir(exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def load(self, name: str, default):
        p = self.path(name)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, name: str, data) -> None:
        # write-then-rename so a crash never leaves a half-written state file
        p = self.path(name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
