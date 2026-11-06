"""summary.csv -> figures: the 3x4 grid of bias panels (one bar per judge) mirroring the paper's figure, plus
cost (mean output tokens per call) vs. metric / Correctness per bias, and the overall utility plot + table. Runs
whose judges were served by our vLLM job also get the two per-bias cost grids with FLOPs per call as the cost."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from .biases import BIASES
from .config import Config

log = logging.getLogger(__name__)

# Validated categorical order (dataviz reference palette, light mode); colors follow the judge, not rank.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF_COLOR = "#d62728"
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"
Y_LABEL = {"robustness": "Robustness Rate", "accuracy": "Accuracy", "error_rate": "Error Rate (lower is better)"}
X_LABEL = {"out_tokens": "Mean output tokens / call", "flops": "Mean FLOPs / call (vLLM)"}
# Effort order within a judge group (one config entry with an effort list); unknown values go last, in config order.
EFFORT_ORDER = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_MARKER = {None: "o", "none": "o", "minimal": "v", "low": "s", "medium": "^", "high": "D", "xhigh": "P",
                 "max": "*"}
MIN_TINT = 0.45  # lowest effort in a group is this share of the group hue (rest white); highest is the full hue
COST_COLUMNS = ["correctness", "calls", "out_tokens"]  # absent in summaries written before they existed


def _ylim(metric: str, values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return (0, 1)
    if metric == "error_rate":
        return (0, max(max(finite) * 1.25, 0.02))
    lo = max(0.0, math.floor((min(finite) - 0.05) * 10) / 10)
    return (min(lo, 0.5), 1.0)


def _effort_rank(judge, judges) -> tuple[int, int]:
    rank = EFFORT_ORDER.index(judge.effort) if judge.effort in EFFORT_ORDER else len(EFFORT_ORDER)
    return rank, judges.index(judge)


def _groups(judges) -> dict[str, list]:
    """{group: its judges in effort order}, groups in config order."""
    groups: dict[str, list] = {}
    for j in judges:
        groups.setdefault(j.group, []).append(j)
    return {g: sorted(m, key=lambda j: _effort_rank(j, judges)) for g, m in groups.items()}


def _tint(color: str, share: float) -> str:
    """`share` of `color`, the rest white."""
    return to_hex(tuple(1 - (1 - c) * share for c in to_rgb(color)))


def _judges_and_colors(cfg: Config, summary: pd.DataFrame):
    """Color follows the group (config entry); within a group, lower effort = lighter tint of its hue."""
    judges = [j for j in cfg.judges if j.label in set(summary["judge"])]
    colors = {}
    for i, members in enumerate(_groups(judges).values()):
        base = next((j.color for j in members if j.color), None) or PALETTE[i % len(PALETTE)]
        for k, j in enumerate(members):
            share = 1.0 if len(members) == 1 else MIN_TINT + (1 - MIN_TINT) * k / (len(members) - 1)
            colors[j.label] = _tint(base, share)
    return judges, colors


def _panel_grid(summary: pd.DataFrame, ncols: int):
    panels = [b for b in BIASES if b in set(summary["bias"])]
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows + 0.6), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    return fig, zip(axes.flat, panels)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(axis="both", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _finish(fig, handles: list, out: Path) -> Path:
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    legend_rows = math.ceil(len(handles) / 7)
    fig.tight_layout(rect=(0, 0.05 * legend_rows + 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_summary(cfg: Config, summary: pd.DataFrame, out: Path, ncols: int = 4) -> Path:
    judges, colors = _judges_and_colors(cfg, summary)
    fig, panels = _panel_grid(summary, ncols)
    for ax, name in panels:
        bias = BIASES[name]
        df = summary[summary["bias"] == name].set_index("judge")
        values = [float(df.loc[j.label, "value"]) if j.label in df.index else math.nan for j in judges]
        ax.bar(range(len(judges)), values, width=0.8, color=[colors[j.label] for j in judges],
               edgecolor="white", linewidth=1.5, zorder=2)

        # Reference: mean consistency rate for robustness panels, mean metric otherwise.
        ref_col = "cr" if bias.metric == "robustness" else "value"
        ref = pd.to_numeric(df[ref_col], errors="coerce").mean()
        if not math.isnan(ref):
            ax.axhline(ref, color=REF_COLOR, linestyle="--", linewidth=1.2, zorder=3)

        ax.set_ylim(*_ylim(bias.metric, values + [ref]))
        _style(ax, bias.title, Y_LABEL[bias.metric])
        ax.set_xticks([])

    handles = [Patch(facecolor=colors[j.label], label=j.label) for j in judges]
    handles.append(Line2D([0], [0], color=REF_COLOR, linestyle="--", label="CR / Avg."))
    return _finish(fig, handles, out)


# --- cost plots ---------------------------------------------------------------------------------

def _with_cost_columns(summary: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COST_COLUMNS if c not in summary.columns]
    if missing:
        log.warning("summary has no %s columns (written by an older version); cost plots stay empty", missing)
        summary = summary.assign(**{c: math.nan for c in missing})
    return summary


def _tick_label(v: float, _=None) -> str:
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= scale:
            return f"{v / scale:g}{suffix}"
    return f"{v:g}"


def _log_x(ax, xs: list[float]) -> None:
    """Log x axis labelled at 1-2-5 steps, so even a narrow token range gets readable ticks."""
    ax.set_xscale("log")
    lo, hi = min(xs), max(xs)
    pad = (hi / lo) ** 0.08 if hi > lo else 1.5
    lo, hi = lo / pad, hi * pad
    ax.set_xlim(lo, hi)
    decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    ticks = [m * 10 ** k for k in decades for m in (1, 2, 5) if lo <= m * 10 ** k <= hi]
    if len(ticks) > 6:  # wide range: decades only
        ticks = [10 ** k for k in decades if lo <= 10 ** k <= hi]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_tick_label))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _cost_scatter(ax, judges, colors, points: dict[str, tuple[float, float]]) -> None:
    """One marker per judge at (tokens, y), shaped by effort; a group's judges are joined in effort order."""
    for members in _groups([j for j in judges if j.label in points]).values():
        if len(members) >= 2:
            ax.plot([points[j.label][0] for j in members], [points[j.label][1] for j in members],
                    color=MUTED, linewidth=1.2, alpha=0.7, zorder=2)
    for j in judges:
        if j.label in points:
            ax.plot(*points[j.label], marker=EFFORT_MARKER.get(j.effort, "o"), markersize=7, color=colors[j.label],
                    markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)


def _points(df: pd.DataFrame, ycol: str, xcol: str = "out_tokens") -> dict[str, tuple[float, float]]:
    """{judge: (xcol, ycol)} for rows where both are known; df is indexed by judge."""
    x, y = pd.to_numeric(df[xcol], errors="coerce"), pd.to_numeric(df[ycol], errors="coerce")
    ok = x.notna() & y.notna() & (x > 0)
    return {label: (float(x[label]), float(y[label])) for label in df.index[ok]}


def _no_data(ax, what: str = "token") -> None:
    ax.text(0.5, 0.5, f"no {what} data", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def _cost_handles(judges, colors) -> list:
    """Legend: one entry per judge (its tint and effort marker), then the group line."""
    handles = [Line2D([0], [0], marker=EFFORT_MARKER.get(j.effort, "o"), markersize=7, color=colors[j.label],
                      markeredgecolor="white", linestyle="none", label=j.label)
               for members in _groups(judges).values() for j in members]
    handles.append(Line2D([0], [0], color=MUTED, linewidth=1.2, label="Same config entry, effort ↑"))
    return handles


def _plot_cost_grid(cfg: Config, summary: pd.DataFrame, out: Path, ycol: str, ncols: int,
                    xcol: str = "out_tokens") -> Path:
    summary = _with_cost_columns(summary)
    judges, colors = _judges_and_colors(cfg, summary)
    fig, panels = _panel_grid(summary, ncols)
    for ax, name in panels:
        bias = BIASES[name]
        points = _points(summary[summary["bias"] == name].set_index("judge"), ycol, xcol)
        _style(ax, bias.title, Y_LABEL[bias.metric] if ycol == "value" else "Correctness")
        ax.set_xlabel(X_LABEL[xcol], fontsize=8, color=MUTED)
        if not points:
            _no_data(ax, "token" if xcol == "out_tokens" else "FLOP")
            continue
        _cost_scatter(ax, judges, colors, points)
        _log_x(ax, [x for x, _ in points.values()])
        ax.set_ylim(*_ylim(bias.metric if ycol == "value" else "accuracy", [y for _, y in points.values()]))
    return _finish(fig, _cost_handles(judges, colors), out)


def plot_cost_performance(cfg: Config, summary: pd.DataFrame, out: Path, ncols: int = 4,
                          xcol: str = "out_tokens") -> Path:
    """Per bias: mean output tokens (or FLOPs, xcol="flops") per call (x) vs. the panel's own metric (y)."""
    return _plot_cost_grid(cfg, summary, out, "value", ncols, xcol)


def plot_correctness(cfg: Config, summary: pd.DataFrame, out: Path, ncols: int = 4, xcol: str = "out_tokens") -> Path:
    """Per bias: mean output tokens (or FLOPs) per call (x) vs. Correctness, the share of samples judged correct (y)."""
    return _plot_cost_grid(cfg, summary, out, "correctness", ncols, xcol)


def utility_table(cfg: Config, summary: pd.DataFrame) -> pd.DataFrame:
    """Per judge: mean/std Correctness over its biases, and mean output tokens per call over all its calls."""
    summary = _with_cost_columns(summary)
    rows = []
    for j in _judges_and_colors(cfg, summary)[0]:
        df = summary[summary["judge"] == j.label]
        corr = pd.to_numeric(df["correctness"], errors="coerce").dropna()
        tok, calls = pd.to_numeric(df["out_tokens"], errors="coerce"), pd.to_numeric(df["calls"], errors="coerce")
        known = tok.notna() & (calls > 0)
        n_calls = int(calls[known].sum())
        rows.append({"judge": j.label, "model": j.model, "effort": j.effort, "n_biases": len(corr),
                     "mean_correctness": corr.mean() if len(corr) else math.nan,
                     "std_correctness": corr.std(ddof=0) if len(corr) else math.nan,
                     "out_tokens": float((tok[known] * calls[known]).sum() / n_calls) if n_calls else math.nan,
                     "calls": n_calls})
    return pd.DataFrame(rows, columns=["judge", "model", "effort", "n_biases", "mean_correctness",
                                       "std_correctness", "out_tokens", "calls"])


def plot_utility(cfg: Config, summary: pd.DataFrame, out: Path, table: pd.DataFrame | None = None) -> Path:
    """Overall: mean output tokens per call (x) vs. mean Correctness across biases (y; bars = std across biases)."""
    t = (utility_table(cfg, summary) if table is None else table).set_index("judge")
    judges, colors = _judges_and_colors(cfg, summary)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    _style(ax, "Mean utility: Correctness vs. cost, all biases", "Mean Correctness across biases")
    ax.set_xlabel(X_LABEL["out_tokens"], fontsize=9, color=MUTED)
    points = _points(t.rename(columns={"mean_correctness": "y"}), "y")
    skipped = sorted(set(t.index) - set(points))
    if skipped:
        log.warning("utility plot skips judges without token or correctness data: %s", skipped)
    if not points:
        _no_data(ax)
        return _finish(fig, _cost_handles(judges, colors), out)
    for label, (x, y) in points.items():
        ax.errorbar(x, y, yerr=t.loc[label, "std_correctness"], color=MUTED, elinewidth=1, capsize=3,
                    linestyle="none", zorder=1)
        ax.annotate(label, (x, y), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK)
    _cost_scatter(ax, judges, colors, points)
    _log_x(ax, [x for x, _ in points.values()])
    ax.set_ylim(*_ylim("accuracy", [y - t.loc[k, "std_correctness"] for k, (_, y) in points.items()]))
    return _finish(fig, _cost_handles(judges, colors), out)


def plot_all(cfg: Config, summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Every figure for one results folder, plus utility.csv (the numbers behind utility.png). The FLOP versions of
    the two cost grids only when some judge has FLOPs (served by our vLLM job)."""
    table = utility_table(cfg, summary)
    table.to_csv(out_dir / "utility.csv", index=False)
    outs = [plot_summary(cfg, summary, out_dir / "bias_grid.png"),
            plot_cost_performance(cfg, summary, out_dir / "cost_vs_performance.png"),
            plot_correctness(cfg, summary, out_dir / "correctness_grid.png"),
            plot_utility(cfg, summary, out_dir / "utility.png", table)]
    if "flops" in summary.columns and pd.to_numeric(summary["flops"], errors="coerce").gt(0).any():
        outs += [plot_cost_performance(cfg, summary, out_dir / "cost_vs_performance_flops.png", xcol="flops"),
                 plot_correctness(cfg, summary, out_dir / "correctness_grid_flops.png", xcol="flops")]
    return outs
