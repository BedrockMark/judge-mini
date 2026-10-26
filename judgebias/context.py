"""Shared run context handed to every bias: data, templates, and a cached, rate-limited `ask`."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from types import ModuleType

from .config import Config, JudgeSpec
from .data import load_dataset
from .llm import ChatClient, ResponseCache, cache_key
from .prompts import load_templates

log = logging.getLogger(__name__)


@dataclass
class Context:
    cfg: Config
    clients: dict[str, ChatClient]
    cache: ResponseCache
    stats: dict[str, int] = field(default_factory=lambda: {"api_calls": 0, "cache_hits": 0, "errors": 0})

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
        self._inflight: dict[str, asyncio.Task] = {}

    @property
    def T(self) -> ModuleType:
        return load_templates(self.cfg.dataset_dir)

    @property
    def judges(self) -> list[JudgeSpec]:
        return self.cfg.judges

    def data(self, name: str) -> list[dict]:
        return load_dataset(self.cfg.dataset_dir, name, self.cfg.samples_per_bias, self.cfg.seed)

    def options(self, bias: str) -> dict:
        return self.cfg.bias_options.get(bias, {})

    async def ask(self, judge: JudgeSpec, prompt: str, run_idx: int = 0) -> str | None:
        """Cached completion; identical concurrent requests are deduplicated. None on failure."""
        key = cache_key(judge.provider, judge.model, judge.effort, judge.params, prompt, run_idx)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            return hit
        if key not in self._inflight:
            self._inflight[key] = asyncio.ensure_future(self._call(key, judge, prompt, run_idx))
        return await self._inflight[key]

    async def _call(self, key: str, judge: JudgeSpec, prompt: str, run_idx: int) -> str | None:
        client = self.clients[judge.provider]
        async with self._sem:
            for attempt in range(3):
                try:
                    text = await client.complete(
                        judge.model, prompt, judge.effort, self.cfg.max_tokens, judge.params
                    )
                    self.stats["api_calls"] += 1
                    self.cache.put(key, text, {"judge": judge.label, "model": judge.model,
                                               "effort": judge.effort, "run_idx": run_idx})
                    return text
                except Exception as e:  # SDKs already retry transient errors; this covers the rest
                    log.warning("%s attempt %d failed: %s", judge.label, attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
        self.stats["errors"] += 1
        return None
