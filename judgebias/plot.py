"""summary.csv -> 3x4 grid of bias panels (one bar per judge), mirroring the paper's figure."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .biases import BIASES
from .config import Config

# Validated categorical order (dataviz reference palette, light mode); colors follow the judge, not rank.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF_COLOR = "#d62728"
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"
Y_LABEL = {"robustness": "Robustness Rate", "accuracy": "Accuracy", "error_rate": "Error Rate"}


def _ylim(metric: str, values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return (0, 1)
    if metric == "error_rate":
        return (0, max(max(finite) * 1.25, 0.02))
    lo = max(0.0, math.floor((min(finite) - 0.05) * 10) / 10)
    return (min(lo, 0.5), 1.0)


def plot_summary(cfg: Config, summary: pd.DataFrame, out: Path, ncols: int = 4) -> Path:
    judges = [j for j in cfg.judges if j.label in set(summary["judge"])]
    colors = {j.label: j.color or PALETTE[i % len(PALETTE)] for i, j in enumerate(judges)}
    panels = [b for b in BIASES if b in set(summary["bias"])]
    nrows = math.ceil(len(panels) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows + 0.6), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax, name in zip(axes.flat, panels):
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
        ax.set_title(bias.title, fontsize=11, color=INK)
        ax.set_ylabel(Y_LABEL[bias.metric], fontsize=9, color=MUTED)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8, colors=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")

    handles = [Patch(facecolor=colors[j.label], label=j.label) for j in judges]
    handles.append(Line2D([0], [0], color=REF_COLOR, linestyle="--", label="CR / Avg."))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    legend_rows = math.ceil(len(handles) / 7)
    fig.tight_layout(rect=(0, 0.05 * legend_rows + 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
