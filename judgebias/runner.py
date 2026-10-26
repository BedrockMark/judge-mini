"""Runs every (judge x bias) pair and writes results/summary.csv plus per-item results/raw/*.jsonl."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import pandas as pd

from .biases import BIASES
from .config import Config, resolve_keys
from .context import Context
from .llm import ResponseCache, build_clients

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["judge", "model", "effort", "bias", "metric", "value", "cr", "n", "n_invalid", "extra"]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def summary_path(cfg: Config) -> Path:
    return cfg.results_dir / "summary.csv"


async def run(cfg: Config, judge_filter: list[str] | None = None, bias_filter: list[str] | None = None) -> pd.DataFrame:
    keys = resolve_keys(cfg)  # fails fast on missing keys, before any call
    judges = [j for j in cfg.judges if not judge_filter or j.label in judge_filter]
    biases = [b for b in cfg.selected_biases() if not bias_filter or b in bias_filter]
    if not judges or not biases:
        raise ValueError("Nothing to run: check --judge / --bias filters")

    cache = ResponseCache(cfg.results_dir / "cache" / "responses.jsonl")
    ctx = Context(cfg, build_clients(cfg, keys), cache)

    async def one(judge, bias_name):
        bias = BIASES[bias_name]
        res = await bias.evaluate(judge, ctx)
        raw = cfg.results_dir / "raw" / slug(judge.label) / f"{bias_name}.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res.records), encoding="utf-8")
        log.info("%-28s %-18s %s=%.3f cr=%.3f n=%d invalid=%d", judge.label, bias_name, bias.metric,
                 res.value, res.cr, res.n, res.n_invalid)
        return {"judge": judge.label, "model": judge.model, "effort": judge.effort, "bias": bias_name,
                "metric": bias.metric, "value": res.value, "cr": res.cr, "n": res.n,
                "n_invalid": res.n_invalid, "extra": json.dumps(res.extra)}

    try:
        rows = await asyncio.gather(*(one(j, b) for j in judges for b in biases))
    finally:
        cache.close()
    log.info("API calls: %(api_calls)d, cache hits: %(cache_hits)d, failed calls: %(errors)d", ctx.stats)

    new = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    path = summary_path(cfg)
    if path.exists():  # merge so partial runs accumulate; new rows replace same (judge, bias)
        old = pd.read_csv(path)
        old = old[~old.set_index(["judge", "bias"]).index.isin(new.set_index(["judge", "bias"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new
