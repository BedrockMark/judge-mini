"""Shared run context handed to every bias: data, templates, and a cached, rate-limited `ask`."""
from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from types import ModuleType

from .config import Config, JudgeSpec
from .data import load_dataset
from .llm import ChatClient, RateLimiter, ResponseCache, cache_key
from .prompts import load_templates

log = logging.getLogger(__name__)

# The bias being evaluated; set by the runner so every `ask` (cache hits too) is billed to (asked judge, bias).
current_bias: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_bias", default=None)


@dataclass
class Usage:
    calls: int = 0  # successful asks
    known: int = 0  # ... of which the output token count is known (older cache entries lack it)
    output_tokens: int = 0

    @property
    def mean_output_tokens(self) -> float:
        return self.output_tokens / self.known if self.known else float("nan")


@dataclass
class Context:
    cfg: Config
    clients: dict[str, ChatClient]
    cache: ResponseCache
    stats: dict[str, int] = field(default_factory=lambda: {"api_calls": 0, "cache_hits": 0, "errors": 0, "rate_limited": 0})
    usage: dict[tuple[str, str | None], Usage] = field(default_factory=dict)  # (judge label, bias) -> tokens

    def __post_init__(self) -> None:
        self._sems: dict[tuple[str, str], asyncio.Semaphore] = {}  # one per (provider, model): each is its own server
        self._limiters = {name: RateLimiter(p.rpm) for name, p in self.cfg.providers.items() if p.rpm}
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
            self._bill(judge, key)
            return hit
        if key not in self._inflight:
            self._inflight[key] = asyncio.ensure_future(self._call(key, judge, prompt, run_idx))
        text = await self._inflight[key]
        if text is not None:
            self._bill(judge, key)
        return text

    def _bill(self, judge: JudgeSpec, key: str) -> None:
        u = self.usage.setdefault((judge.label, current_bias.get()), Usage())
        u.calls += 1
        tokens = self.cache.output_tokens(key)
        if tokens is not None:
            u.known += 1
            u.output_tokens += tokens

    async def _call(self, key: str, judge: JudgeSpec, prompt: str, run_idx: int) -> str | None:
        client = self.clients[judge.provider]
        sem = self._sems.setdefault((judge.provider, judge.model), asyncio.Semaphore(self.cfg.concurrency))
        limiter = self._limiters.get(judge.provider)
        async with sem:
            attempt = throttled = 0
            while attempt < 3:
                if limiter:
                    await limiter.acquire()
                try:
                    text = await client.complete(
                        judge.model, prompt, judge.effort, self.cfg.max_tokens, judge.params
                    )
                    self.stats["api_calls"] += 1
                    self.cache.put(key, text, {"judge": judge.label, "model": judge.model,
                                               "effort": judge.effort, "run_idx": run_idx},
                                   getattr(text, "output_tokens", None))
                    return str(text)
                except Exception as e:  # SDKs already retry transient errors; this covers the rest
                    if getattr(e, "status_code", None) == 429 and throttled < 20:
                        # quota window is 60 s and shared by the key (other runs too): wait it out, not an attempt
                        throttled += 1
                        self.stats["rate_limited"] += 1
                        log.warning("%s rate-limited (429), waiting 30s", judge.label)
                        await asyncio.sleep(30)
                        continue
                    attempt += 1
                    log.warning("%s attempt %d failed: %s", judge.label, attempt, e)
                    await asyncio.sleep(2 ** attempt)
        self.stats["errors"] += 1
        return None
