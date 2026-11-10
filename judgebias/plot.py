"""summary.csv -> figures: the 3x4 grids of bias panels (one bar per judge, with its 95% interval) of each bias's
metric, mirroring the paper's figure, and of Correctness; cost (mean output tokens per call) vs. metric / Correctness
per bias; and the overall utility plot + table. Runs whose judges were served by our vLLM job also get the two
per-bias cost grids with FLOPs per call as the cost.

Judges, groups and efforts come from the summary itself (its judge / group / model / effort columns); the config is
only consulted for a judge's `color`."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
from .config import Config, effort_rank
from .metrics import NAN_CI, wilson

log = logging.getLogger(__name__)

# Validated categorical order (dataviz reference palette, light mode); colors follow the judge, not rank.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF_COLOR = "#d62728"
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"
Y_LABEL = {"robustness": "Robustness Rate", "accuracy": "Accuracy", "error_rate": "Error Rate (lower is better)"}
X_LABEL = {"out_tokens": "Mean output tokens / call", "flops": "Mean FLOPs / call (vLLM)"}
EFFORT_MARKER = {None: "o", "none": "o", "off": "o", "minimal": "v", "low": "s", "medium": "^", "high": "D",
                 "on": "D", "xhigh": "P", "max": "*"}
MIN_TINT = 0.45  # lowest effort in a group is this share of the group hue (rest white); highest is the full hue
ERR_KW = {"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "alpha": 0.8}


@dataclass(frozen=True)
class Judge:
    label: str
    group: str  # judges sharing a hue, joined by a line in the cost plots (one config entry with an effort list)
    model: str
    effort: str | None
    color: str | None = None  # config override of the group hue
    kind: str | None = None  # compare: the group as named in its own run (`group` is then "<group> [<run code>]")


def effort_of(value) -> str | None:
    """A summary's effort cell (NaN for a judge without one) as the effort string, or None."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) or value == "" else str(value)


def judges_of(summary: pd.DataFrame, cfg: Config | None = None) -> list[Judge]:
    """The judges a summary holds, grouped (groups in order of appearance, efforts low to high within a group)."""
    colors = {j.label: j.color for j in cfg.judges if j.color} if cfg else {}
    seen = summary.drop_duplicates("judge")
    judges = [Judge(r.judge, r.group, r.model, effort_of(r.effort), colors.get(r.judge)) for r in seen.itertuples()]
    return [j for members in groups_of(judges).values() for j in members]


def groups_of(judges: list[Judge]) -> dict[str, list[Judge]]:
    """{group: its judges in effort order}, groups in order of appearance."""
    groups: dict[str, list[Judge]] = {}
    for j in judges:
        groups.setdefault(j.group, []).append(j)
    return {g: sorted(m, key=lambda j: (effort_rank(j.effort), judges.index(j))) for g, m in groups.items()}


def _tint(color: str, share: float) -> str:
    """`share` of `color`, the rest white."""
    return to_hex(tuple(1 - (1 - c) * share for c in to_rgb(color)))


def tints(members: list[Judge], base: str) -> dict[str, str]:
    """One tint of `base` per judge of a group: lowest effort lightest, highest the full hue."""
    return {j.label: _tint(base, 1.0 if len(members) == 1 else MIN_TINT + (1 - MIN_TINT) * k / (len(members) - 1))
            for k, j in enumerate(members)}


def colors_of(judges: list[Judge]) -> dict[str, str]:
    """Color follows the group (its i-th palette hue, or a member's config color); within a group, lower effort =
    lighter tint."""
    colors = {}
    for i, members in enumerate(groups_of(judges).values()):
        base = next((j.color for j in members if j.color), None) or PALETTE[i % len(PALETTE)]
        colors.update(tints(members, base))
    return colors


def _ylim(metric: str, values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return (0, 1)
    if metric == "error_rate":
        return (0, max(max(finite) * 1.15, 0.02))
    lo = max(0.0, math.floor((min(finite) - 0.05) * 10) / 10)
    return (min(lo, 0.5), 1.0)


def interval(df: pd.DataFrame, ycol: str) -> tuple[pd.Series, pd.Series]:
    """95% interval of ycol per row: the stored one for the panel metric, Wilson from the counts for Correctness."""
    if ycol == "value":
        return pd.to_numeric(df["value_lo"], errors="coerce"), pd.to_numeric(df["value_hi"], errors="coerce")
    bounds = [wilson(int(k), int(n)) if n > 0 else NAN_CI
              for k, n in zip(pd.to_numeric(df["n_correct"], errors="coerce").fillna(0),
                              pd.to_numeric(df["samples"], errors="coerce").fillna(0))]
    return pd.Series([b[0] for b in bounds], index=df.index), pd.Series([b[1] for b in bounds], index=df.index)


def yerr(values: list[float], lo: list[float], hi: list[float]) -> list[list[float]]:
    """matplotlib's asymmetric error bars (NaN -> 0 so the bar still draws)."""
    down = [0.0 if math.isnan(v) or math.isnan(x) else v - x for v, x in zip(values, lo)]
    up = [0.0 if math.isnan(v) or math.isnan(x) else x - v for v, x in zip(values, hi)]
    return [down, up]


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


def _finish(fig, handles: list, out: Path, ncol: int = 7) -> Path:
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), ncol), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    legend_rows = math.ceil(len(handles) / ncol)
    fig.tight_layout(rect=(0, 0.05 * legend_rows + 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _reference(ax, df: pd.DataFrame, bias, ycol: str) -> float:
    """Dashed line: mean consistency rate for robustness panels, mean of the plotted column otherwise."""
    ref_col = "cr" if ycol == "value" and bias.metric == "robustness" else ycol
    ref = pd.to_numeric(df[ref_col], errors="coerce").mean()
    if not math.isnan(ref):
        ax.axhline(ref, color=REF_COLOR, linestyle="--", linewidth=1.2, zorder=3)
    return ref


def _ref_handle(ycol: str) -> Line2D:
    return Line2D([0], [0], color=REF_COLOR, linestyle="--", label="CR / Avg." if ycol == "value" else "Avg.")


def _bar_grid(summary: pd.DataFrame, judges: list[Judge], colors: dict[str, str], out: Path, ycol: str,
              ncols: int) -> Path:
    """One bar (with its 95% interval) per judge per bias panel: ycol "value" = the panel's own metric,
    "correctness" = Correctness. Bars are grouped by judge group, effort ascending within each."""
    fig, panels = _panel_grid(summary, ncols)
    for ax, name in panels:
        bias = BIASES[name]
        df = summary[summary["bias"] == name].set_index("judge")
        lo, hi = interval(df, ycol)
        pick = lambda s, j: float(s[j.label]) if j.label in df.index else math.nan
        values = [pick(df[ycol], j) for j in judges]
        ax.bar(range(len(judges)), values, width=0.8, color=[colors[j.label] for j in judges], edgecolor="white",
               linewidth=1.5, zorder=2, yerr=yerr(values, [pick(lo, j) for j in judges], [pick(hi, j) for j in judges]),
               error_kw=ERR_KW)
        ref = _reference(ax, df, bias, ycol)
        ax.set_ylim(*_ylim(bias.metric if ycol == "value" else "accuracy", values + [ref, *lo, *hi]))
        _style(ax, bias.title, Y_LABEL[bias.metric] if ycol == "value" else "Correctness")
        ax.set_xticks([])
    handles = [Patch(facecolor=colors[j.label], label=j.label) for j in judges]
    handles += [Line2D([0], [0], color=MUTED, linewidth=0.8, label="95% CI"), _ref_handle(ycol)]
    return _finish(fig, handles, out)


# --- cost plots ---------------------------------------------------------------------------------

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


Point = tuple[float, float, float, float]  # x, y, y_lo, y_hi


def _points(df: pd.DataFrame, ycol: str, xcol: str = "out_tokens") -> dict[str, Point]:
    """{judge: (x, y, y_lo, y_hi)} for rows where x and y are known; df is indexed by judge."""
    x, y = pd.to_numeric(df[xcol], errors="coerce"), pd.to_numeric(df[ycol], errors="coerce")
    lo, hi = interval(df, ycol)
    ok = x.notna() & y.notna() & (x > 0)
    return {label: (float(x[label]), float(y[label]), float(lo[label]), float(hi[label])) for label in df.index[ok]}


def _cost_scatter(ax, judges: list[Judge], colors: dict[str, str], points: dict[str, Point]) -> None:
    """One marker per judge at (cost, y) with its 95% interval, shaped by effort; a group's judges are joined in
    effort order."""
    for members in groups_of([j for j in judges if j.label in points]).values():
        if len(members) >= 2:
            ax.plot([points[j.label][0] for j in members], [points[j.label][1] for j in members],
                    color=MUTED, linewidth=1.2, alpha=0.7, zorder=2)
    for j in judges:
        if j.label in points:
            x, y, lo, hi = points[j.label]
            ax.errorbar(x, y, yerr=yerr([y], [lo], [hi]), color=colors[j.label], elinewidth=0.8, capsize=0,
                        linestyle="none", alpha=0.6, zorder=2)
            ax.plot(x, y, marker=EFFORT_MARKER.get(j.effort, "o"), markersize=7, color=colors[j.label],
                    markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)


def _no_data(ax, what: str = "token") -> None:
    ax.text(0.5, 0.5, f"no {what} data", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def _cost_handles(judges: list[Judge], colors: dict[str, str]) -> list:
    """Legend: one entry per judge (its tint and effort marker), then the group line."""
    handles = [Line2D([0], [0], marker=EFFORT_MARKER.get(j.effort, "o"), markersize=7, color=colors[j.label],
                      markeredgecolor="white", linestyle="none", label=j.label) for j in judges]
    handles.append(Line2D([0], [0], color=MUTED, linewidth=1.2, label="Same group, effort ↑ (bars: 95% CI)"))
    return handles


def _cost_grid(summary: pd.DataFrame, judges: list[Judge], colors: dict[str, str], out: Path, ycol: str, ncols: int,
               xcol: str = "out_tokens") -> Path:
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
        _log_x(ax, [p[0] for p in points.values()])
        ax.set_ylim(*_ylim(bias.metric if ycol == "value" else "accuracy",
                           [v for p in points.values() for v in p[1:]]))
    return _finish(fig, _cost_handles(judges, colors), out)


def utility_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per judge: mean/std Correctness over its biases, and mean output tokens per call over all its calls."""
    rows = []
    for j in judges_of(summary):
        df = summary[summary["judge"] == j.label]
        corr = pd.to_numeric(df["correctness"], errors="coerce").dropna()
        tok, calls = pd.to_numeric(df["out_tokens"], errors="coerce"), pd.to_numeric(df["calls"], errors="coerce")
        known = tok.notna() & (calls > 0)
        n_calls = int(calls[known].sum())
        rows.append({"judge": j.label, "group": j.group, "model": j.model, "effort": j.effort, "n_biases": len(corr),
                     "mean_correctness": corr.mean() if len(corr) else math.nan,
                     "std_correctness": corr.std(ddof=0) if len(corr) else math.nan,
                     "out_tokens": float((tok[known] * calls[known]).sum() / n_calls) if n_calls else math.nan,
                     "calls": n_calls})
    return pd.DataFrame(rows, columns=["judge", "group", "model", "effort", "n_biases", "mean_correctness",
                                       "std_correctness", "out_tokens", "calls"])


def plot_utility(summary: pd.DataFrame, judges: list[Judge], colors: dict[str, str], out: Path,
                 table: pd.DataFrame) -> Path:
    """Overall: mean output tokens per call (x) vs. mean Correctness across biases (y; bars = std across biases,
    i.e. how unevenly the judge does across panels, not sampling error)."""
    t = table.set_index("judge")
    fig, ax = plt.subplots(figsize=(7, 5.2))
    _style(ax, "Mean utility: Correctness vs. cost, all biases", "Mean Correctness across biases")
    ax.set_xlabel(X_LABEL["out_tokens"], fontsize=9, color=MUTED)
    x, y = pd.to_numeric(t["out_tokens"], errors="coerce"), pd.to_numeric(t["mean_correctness"], errors="coerce")
    sd = pd.to_numeric(t["std_correctness"], errors="coerce").fillna(0)
    ok = x.notna() & y.notna() & (x > 0)
    points = {k: (float(x[k]), float(y[k]), float(y[k] - sd[k]), float(y[k] + sd[k])) for k in t.index[ok]}
    if skipped := sorted(set(t.index) - set(points)):
        log.warning("utility plot skips judges without token or correctness data: %s", skipped)
    handles = _cost_handles(judges, colors)
    handles[-1].set_label("Same group, effort ↑ (bars: ± std across biases)")
    if not points:
        _no_data(ax)
        return _finish(fig, handles, out)
    for label, (px, py, _, _) in points.items():
        ax.annotate(label, (px, py), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK)
    _cost_scatter(ax, judges, colors, points)
    _log_x(ax, [p[0] for p in points.values()])
    ax.set_ylim(*_ylim("accuracy", [p[2] for p in points.values()]))
    return _finish(fig, handles, out)


def plot_all(cfg: Config, summary: pd.DataFrame, out_dir: Path, ncols: int = 4) -> list[Path]:
    """Every figure for one results folder, plus utility.csv (the numbers behind utility.png). The FLOP versions of
    the two cost grids only when some judge has FLOPs (served by our vLLM job)."""
    judges = judges_of(summary, cfg)
    colors = colors_of(judges)
    table = utility_table(summary)
    table.to_csv(out_dir / "utility.csv", index=False)
    outs = [_bar_grid(summary, judges, colors, out_dir / "bias_grid.png", "value", ncols),
            _cost_grid(summary, judges, colors, out_dir / "cost_vs_performance.png", "value", ncols),
            _bar_grid(summary, judges, colors, out_dir / "correctness_grid.png", "correctness", ncols),
            _cost_grid(summary, judges, colors, out_dir / "cost_vs_correctness.png", "correctness", ncols),
            plot_utility(summary, judges, colors, out_dir / "utility.png", table)]
    if pd.to_numeric(summary["flops"], errors="coerce").gt(0).any():
        outs += [_cost_grid(summary, judges, colors, out_dir / "cost_vs_performance_flops.png", "value", ncols, "flops"),
                 _cost_grid(summary, judges, colors, out_dir / "cost_vs_correctness_flops.png", "correctness", ncols,
                            "flops")]
    return outs
