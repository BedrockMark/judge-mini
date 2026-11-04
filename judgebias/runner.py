"""Runs every (judge x bias) pair into a fresh results/<YYYYmmdd_HHMMSS>/ folder: summary.csv, per-item
raw/*.jsonl and per-call examples/<bias>/<judge>/{correct,wrong,invalid}.md. Only results/cache/ is shared."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .biases import BIASES
from .config import Config, resolve_keys
from .context import Context
from .examples import write_examples
from .llm import ResponseCache, build_clients

log = logging.getLogger(__name__)

SUMMARY_COLUMNS = ["judge", "model", "effort", "bias", "metric", "value", "cr", "n", "n_invalid", "extra"]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}(_\d+)?$")


def new_output_dir(cfg: Config) -> Path:
    """A new, empty results/<YYYYmmdd_HHMMSS>[_n]/ folder; earlier outputs are never overwritten."""
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1000):
        out = cfg.results_dir / (stamp if n == 0 else f"{stamp}_{n}")
        try:
            out.mkdir()
            return out
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create an output folder under {cfg.results_dir}")


def latest_output_dir(cfg: Config) -> Path:
    """Newest timestamped folder that has a summary.csv."""
    dirs = sorted(d for d in cfg.results_dir.glob("*") if RUN_DIR_RE.match(d.name) and (d / "summary.csv").exists())
    if not dirs:
        raise FileNotFoundError(f"No results/<timestamp>/summary.csv under {cfg.results_dir}; run first")
    return dirs[-1]


def merge_summaries(dirs: list[Path]) -> pd.DataFrame:
    """Concatenate several runs' summaries; a later folder's (judge, bias) row replaces an earlier one."""
    df = pd.concat([pd.read_csv(d / "summary.csv") for d in dirs], ignore_index=True)
    return df.drop_duplicates(["judge", "bias"], keep="last").reset_index(drop=True)


async def run(cfg: Config, out_dir: Path, judge_filter: list[str] | None = None,
              bias_filter: list[str] | None = None) -> pd.DataFrame:
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
        raw = out_dir / "raw" / slug(judge.label) / f"{bias_name}.jsonl"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res.records), encoding="utf-8")
        write_examples(out_dir / "examples" / bias_name / slug(judge.label), judge, bias, res.samples)
        log.info("%-28s %-18s %s=%.3f cr=%.3f n=%d invalid=%d", judge.label, bias_name, bias.metric,
                 res.value, res.cr, res.n, res.n_invalid)
        return {"judge": judge.label, "model": judge.model, "effort": judge.effort, "bias": bias_name,
                "metric": bias.metric, "value": res.value, "cr": res.cr, "n": res.n,
                "n_invalid": res.n_invalid, "extra": json.dumps(res.extra)}

    try:
        rows = await asyncio.gather(*(one(j, b) for j in judges for b in biases))
    finally:
        cache.close()
    log.info("API calls: %(api_calls)d, cache hits: %(cache_hits)d, failed calls: %(errors)d, 429s: %(rate_limited)d", ctx.stats)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary
