"""Append-only JSONL response cache. Only successful responses are stored, so failures are retried next run.
Each record also keeps the response's output and prompt token counts (missing in entries written before they were
recorded) and the thinking text, when the API returned one. Every record is flushed as written, so a run killed midway leaves the calls it finished (see runner.run)."""
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
        self._tokens: dict[str, int | None] = {}
        self._prompt_tokens: dict[str, int | None] = {}
        self._reasoning: dict[str, str] = {}
        if path.exists():
            raw = path.read_bytes()
            if not raw.endswith(b"\n"):  # killed mid-write (e.g. a job's time limit): drop the partial last record
                raw = raw[:raw.rfind(b"\n") + 1]
                path.write_bytes(raw)
            for line in raw.decode("utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["response"]
                    self._tokens[rec["key"]] = rec.get("output_tokens")
                    self._prompt_tokens[rec["key"]] = rec.get("prompt_tokens")
                    if rec.get("reasoning"):
                        self._reasoning[rec["key"]] = rec["reasoning"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def output_tokens(self, key: str) -> int | None:
        return self._tokens.get(key)

    def prompt_tokens(self, key: str) -> int | None:
        return self._prompt_tokens.get(key)

    def reasoning(self, key: str) -> str | None:
        return self._reasoning.get(key)

    def put(self, key: str, response: str, meta: dict, output_tokens: int | None = None,
            prompt_tokens: int | None = None, reasoning: str | None = None) -> None:
        self._data[key] = str(response)
        self._tokens[key] = output_tokens
        self._prompt_tokens[key] = prompt_tokens
        rec = {"key": key, **meta, "output_tokens": output_tokens, "prompt_tokens": prompt_tokens,
               "response": str(response)}
        if reasoning:
            self._reasoning[key] = reasoning
            rec["reasoning"] = reasoning
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
