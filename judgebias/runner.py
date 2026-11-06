"""Runs every (judge x bias) pair into a fresh results/<YYYYmmdd_HHMMSS>/ folder: summary.csv and examples.zip
(per-call examples/<bias>/<judge>/{correct,wrong,invalid}.md; the unzipped folder is deleted once zipped). Each run starts with an empty response cache (only deduplicates calls within the run) and deletes it
when done, so nothing is reused from earlier runs."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from .biases import BIASES
from .config import Config, require_dataset, resolve_keys
from .context import Context, current_bias
from .examples import write_examples
from .llm import ResponseCache, build_clients

log = logging.getLogger(__name__)

# samples = every judged call classified correct/wrong/invalid (examples/ shows only a few of them; n counts dataset
# rows), correctness = n_correct / samples; out_tokens = mean output tokens (thinking included) per call, over calls
# whose usage is known (NaN if none, e.g. all answers came from an older cache).
SUMMARY_COLUMNS = ["judge", "group", "model", "effort", "bias", "metric", "value", "cr", "n", "n_invalid", "samples",
                   "n_correct", "n_wrong", "correctness", "calls", "out_tokens", "extra"]


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


def given_output_dir(path: Path) -> Path:
    """A caller-chosen output folder (e.g. one part of a split job); it must not exist yet, be empty, or hold only the
    config snapshot a job script put there at submission."""
    path.mkdir(parents=True, exist_ok=True)
    if any(p.name != "config.yaml" for p in path.iterdir()):
        raise FileExistsError(f"Output folder {path} is not empty")
    return path


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


def zip_examples(out_dir: Path) -> None:
    """out_dir/examples/ -> out_dir/examples.zip (holding the examples/ folder itself); the folder is then deleted."""
    shutil.make_archive(str(out_dir / "examples"), "zip", root_dir=out_dir, base_dir="examples")
    shutil.rmtree(out_dir / "examples")


def merge_details(dirs: list[Path], out_dir: Path) -> None:
    """Merge each run's examples.zip (or an older run's examples/ folder) into out_dir/examples.zip, later folder
    winning per file. Files are per (judge, bias), so this matches merge_summaries when the runs split the biases or
    judges. Older runs' raw/ and raw.zip are not carried over."""
    found = False
    for d in dirs:
        if (d / "examples.zip").exists():
            shutil.unpack_archive(d / "examples.zip", out_dir)  # the zip holds the examples/ folder itself
        elif (d / "examples").is_dir():
            shutil.copytree(d / "examples", out_dir / "examples", dirs_exist_ok=True)
        else:
            continue
        found = True
    if found:
        zip_examples(out_dir)


async def run(cfg: Config, out_dir: Path, judge_filter: list[str] | None = None,
              bias_filter: list[str] | None = None) -> pd.DataFrame:
    keys = resolve_keys(cfg)  # fails fast on missing keys, before any call
    require_dataset(cfg)
    judges = [j for j in cfg.judges if not judge_filter or j.label in judge_filter]
    biases = [b for b in cfg.selected_biases() if not bias_filter or b in bias_filter]
    if not judges or not biases:
        raise ValueError("Nothing to run: check --judge / --bias filters")

    cache_dir = out_dir / "cache"  # out_dir is new, so this cache starts empty
    cache = ResponseCache(cache_dir / "responses.jsonl")
    ctx = Context(cfg, build_clients(cfg, keys), cache)

    async def one(judge, bias_name):
        current_bias.set(bias_name)  # gather runs each in its own task, so this tags only this pair's calls
        bias = BIASES[bias_name]
        res = await bias.evaluate(judge, ctx)
        write_examples(out_dir / "examples" / bias_name / slug(judge.label), judge, bias, res.samples)
        log.info("%-28s %-18s %s=%.3f cr=%.3f n=%d invalid=%d", judge.label, bias_name, bias.metric,
                 res.value, res.cr, res.n, res.n_invalid)
        counts = Counter(x["status"] for x in res.samples)
        return {"judge": judge.label, "group": judge.group, "model": judge.model, "effort": judge.effort, "bias": bias_name,
                "metric": bias.metric, "value": res.value, "cr": res.cr, "n": res.n,
                "n_invalid": res.n_invalid, "samples": len(res.samples), "n_correct": counts["correct"],
                "n_wrong": counts["wrong"],
                "correctness": counts["correct"] / len(res.samples) if res.samples else math.nan,
                "extra": json.dumps(res.extra)}

    try:
        rows = await asyncio.gather(*(one(j, b) for j in judges for b in biases))
    finally:
        cache.close()
        shutil.rmtree(cache_dir, ignore_errors=True)
    for row in rows:  # after gather: Self-Enhancement bills other judges' scoring calls to their own rows
        u = ctx.usage.get((row["judge"], row["bias"]))
        row["calls"], row["out_tokens"] = (u.known, u.mean_output_tokens) if u else (0, math.nan)
        if u and u.known < u.calls:
            log.warning("%s / %s: %d of %d calls have no token count (older cache entries); out_tokens uses the rest",
                        row["judge"], row["bias"], u.calls - u.known, u.calls)
    log.info("API calls: %(api_calls)d, cache hits: %(cache_hits)d, failed calls: %(errors)d, 429s: %(rate_limited)d", ctx.stats)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary.to_csv(out_dir / "summary.csv", index=False)
    zip_examples(out_dir)
    return summary
