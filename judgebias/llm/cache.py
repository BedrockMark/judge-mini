"""Append-only JSONL response cache. Only successful responses are stored, so failures are retried next run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(provider: str, model: str, effort: str | None, params: dict, prompt: str, run_idx: int) -> str:
    blob = json.dumps([provider, model, effort, params, prompt, run_idx], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["response"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, response: str, meta: dict) -> None:
        self._data[key] = response
        self._fh.write(json.dumps({"key": key, **meta, "response": response}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
