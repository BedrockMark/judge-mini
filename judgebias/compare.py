"""Compare finished runs without calling any model: several results folders' summary.csv -> six figures.

Each (folder, group) is one model. In the two bar grids, bars are clustered by effort tier: a model's efforts, sorted
low to high, fill tiers 1, 2, ... from the bottom, and a model with fewer efforts repeats its highest one in the upper
tiers (hatched bars). E.g. A {none, high} and B {low, medium, high} give tiers [A none, B low], [A high, B medium],
[A high (repeat), B high]. The cost grids (output tokens, and FLOPs for vLLM-served judges) overlay every model's judges in one scatter per bias."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .biases import BIASES
from .plot import (MIN_TINT, MUTED, PALETTE, REF_COLOR, X_LABEL, Y_LABEL, _cost_handles, _cost_scatter, _finish,
                   _log_x, _no_data, _panel_grid, _points, _style, _tint, _with_cost_columns, _ylim)

log = logging.getLogger(__name__)

# Low-to-high effort order used to stack a model's efforts into tiers; unknown values go last, in appearance order.
EFFORT_ORDER = [None, "none", "off", "minimal", "low", "medium", "high", "on", "xhigh", "max"]
COPY_HATCH = "////"


def _effort(v) -> str | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or v == "" else str(v)


def _group(row) -> str:
    """Summaries written before the `group` column: the judge label without its " (<effort>)" suffix."""
    g = row.get("group")
    if isinstance(g, str) and g:
        return g
    m = re.fullmatch(r"(.*) \((.+)\)", str(row["judge"]))
    return m.group(1) if m and m.group(2) == row["effort"] else str(row["judge"])


def load_runs(dirs: list[Path]) -> pd.DataFrame:
    """All summaries in one frame. `model` = the group, suffixed with " [<folder>]" when two folders share a group;
    `judge` = "<model> (<effort>)", unique across folders."""
    names = [d.name for d in dirs]
    frames = []
    for i, d in enumerate(dirs):
        df = _with_cost_columns(pd.read_csv(d / "summary.csv"))
        if "flops" not in df.columns:  # summaries written before FLOPs, or without any vLLM judge
            df["flops"] = math.nan
        df["effort"] = df["effort"].map(_effort).astype(object)
        df["group"] = df.apply(_group, axis=1)
        df["source"] = names[i] if names.count(names[i]) == 1 else f"{names[i]}#{i + 1}"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    shared = df.groupby("group")["source"].nunique()
    df["model"] = [g if shared[g] == 1 else f"{g} [{s}]" for g, s in zip(df["group"], df["source"])]
    df["judge"] = [m if e is None else f"{m} ({e})" for m, e in zip(df["model"], df["effort"])]
    return df.drop_duplicates(["judge", "bias"], keep="last").reset_index(drop=True)


def _rank(effort: str | None, seen: list) -> tuple[int, int]:
    return (EFFORT_ORDER.index(effort), 0) if effort in EFFORT_ORDER else (len(EFFORT_ORDER), seen.index(effort))


def _models(df: pd.DataFrame) -> dict[str, list]:
    """{model: its judges (label, group, effort) low to high}, models in folder order; each model gets a hue and its
    efforts are tints of it (lowest lightest), as in the single-run figures."""
    models: dict[str, list] = {}
    for model, sub in df.groupby("model", sort=False):
        seen = list(dict.fromkeys(sub["effort"]))
        efforts = sorted(seen, key=lambda e: _rank(e, seen))
        models[model] = [SimpleNamespace(label=model if e is None else f"{model} ({e})", group=model, effort=e)
                         for e in efforts]
    return models


def _colors(models: dict[str, list]) -> dict[str, str]:
    colors = {}
    for i, members in enumerate(models.values()):
        for k, j in enumerate(members):
            share = 1.0 if len(members) == 1 else MIN_TINT + (1 - MIN_TINT) * k / (len(members) - 1)
            colors[j.label] = _tint(PALETTE[i % len(PALETTE)], share)
    return colors


def tiers(models: dict[str, list]) -> list[list[tuple]]:
    """tiers[t] = [(judge, is_repeat) per model]: bottom-aligned, a model's highest effort repeated upward."""
    n = max(len(m) for m in models.values())
    return [[(m[min(t, len(m) - 1)], t >= len(m)) for m in models.values()] for t in range(n)]


def _with_data(df: pd.DataFrame, cols: list[str], out: Path) -> pd.DataFrame:
    """Only models with some row where all `cols` are known (older summaries lack correctness / token columns)."""
    known = df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    keep = set(df.loc[known, "model"])
    if not keep:
        log.warning("%s skipped: no model has %s data", out.name, "/".join(cols))
    elif dropped := sorted(set(df["model"]) - keep):
        log.warning("%s leaves out models without %s data: %s", out.name, "/".join(cols), dropped)
    return df[df["model"].isin(keep)]


def _bar_grid(df: pd.DataFrame, out: Path, ycol: str, ncols: int) -> Path | None:
    df = _with_data(df, [ycol], out)
    if df.empty:
        return None
    models = _models(df)
    colors, layers = _colors(models), tiers(models)
    width = 0.8 / len(models)
    fig, panels = _panel_grid(df, ncols)
    for ax, name in panels:
        bias = BIASES[name]
        metric = bias.metric if ycol == "value" else "accuracy"
        sub = df[df["bias"] == name].set_index("judge")
        values = []
        for t, tier in enumerate(layers):
            for k, (j, repeat) in enumerate(tier):
                v = float(sub.loc[j.label, ycol]) if j.label in sub.index else math.nan
                values.append(v)
                ax.bar(t - 0.4 + width * (k + 0.5), v, width=width, color=colors[j.label],
                       hatch=COPY_HATCH if repeat else None, edgecolor="white", linewidth=1, zorder=2)
        ref_col = "cr" if ycol == "value" and bias.metric == "robustness" else ycol
        ref = pd.to_numeric(sub[ref_col], errors="coerce").mean()
        if not math.isnan(ref):
            ax.axhline(ref, color=REF_COLOR, linestyle="--", linewidth=1.2, zorder=3)
        ax.set_ylim(*_ylim(metric, values + [ref]))
        _style(ax, bias.title, Y_LABEL[bias.metric] if ycol == "value" else "Correctness")
        ax.set_xticks(range(len(layers)), [f"T{t + 1}" for t in range(len(layers))])
        ax.set_xlabel("Effort tier (low → high)", fontsize=8, color=MUTED)

    handles = [Patch(facecolor=colors[members[-1].label],
                     label=f"{model}: {' < '.join(j.effort or 'default' for j in members)}")
               for model, members in models.items()]
    handles.append(Patch(facecolor="#c3c2b7", hatch=COPY_HATCH, edgecolor="white", label="highest effort, repeated"))
    handles.append(Line2D([0], [0], color=REF_COLOR, linestyle="--", label="CR / Avg." if ycol == "value" else "Avg."))
    return _finish(fig, handles, out, ncol=3)


def _cost_grid(df: pd.DataFrame, out: Path, ycol: str, ncols: int, xcol: str = "out_tokens") -> Path | None:
    """xcol = "out_tokens" (mean output tokens / call) or "flops" (mean FLOPs / call, vLLM-served judges only)."""
    df = _with_data(df, [ycol, xcol], out)
    if df.empty:
        return None
    models = _models(df)
    judges = [j for members in models.values() for j in members]
    colors = _colors(models)
    fig, panels = _panel_grid(df, ncols)
    for ax, name in panels:
        bias = BIASES[name]
        points = _points(df[df["bias"] == name].set_index("judge"), ycol, xcol)
        _style(ax, bias.title, Y_LABEL[bias.metric] if ycol == "value" else "Correctness")
        ax.set_xlabel(X_LABEL[xcol], fontsize=8, color=MUTED)
        if not points:
            _no_data(ax, "token" if xcol == "out_tokens" else "FLOP")
            continue
        _cost_scatter(ax, judges, colors, points)
        _log_x(ax, [x for x, _ in points.values()])
        ax.set_ylim(*_ylim(bias.metric if ycol == "value" else "accuracy", [y for _, y in points.values()]))
    return _finish(fig, _cost_handles(judges, colors), out, ncol=4)


def compare_all(dirs: list[Path], out_dir: Path, ncols: int = 4) -> list[Path]:
    """The comparison figures for these run folders, plus the combined summary they were drawn from: the two bar
    grids, the two cost grids, and the two cost grids with FLOPs as the cost (judges served by our vLLM job). A
    figure no model has data for is skipped."""
    df = load_runs(dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    outs = [_bar_grid(df, out_dir / "bias_grid.png", "value", ncols),
            _bar_grid(df, out_dir / "correctness_grid.png", "correctness", ncols),
            _cost_grid(df, out_dir / "cost_vs_performance.png", "value", ncols),
            _cost_grid(df, out_dir / "cost_vs_correctness.png", "correctness", ncols),
            _cost_grid(df, out_dir / "cost_vs_performance_flops.png", "value", ncols, "flops"),
            _cost_grid(df, out_dir / "cost_vs_correctness_flops.png", "correctness", ncols, "flops")]
    return [p for p in outs if p is not None]
