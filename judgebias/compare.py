"""Compare finished runs without calling any model: several downloaded results' summary.csv -> seven figures.

A source is a folder or a zip (e.g. from `python -m judgebias get`) holding one run's summary.csv: at its top, in a
single wrapping subfolder, or a vllm_split.sh folder not merged yet, whose part<i>/ summaries are merged (a later
part winning on the same judge x bias).

Each (source, group) is one model; a group in several sources is told apart by the run code its summary.csv records.
In the two bar grids, bars are clustered by effort tier: a model's efforts, sorted low to high, fill tiers 1, 2, ...
from the bottom, and a model with fewer efforts repeats its highest one in the upper tiers (hatched bars). E.g. A
{none, high} and B {low, medium, high} give tiers [A none, B low], [A high, B medium], [A high (repeat), B high]. The
cost grids (output tokens, and FLOPs for vLLM-served judges) instead pool a group's runs into one model, named
"<group> [<run codes>]": per effort, the mean cost and metric across its runs, with ±1 SD bars."""
from __future__ import annotations

import logging
import math
import re
import zipfile
from colorsys import hls_to_rgb, rgb_to_hls
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch

from .config import effort_rank
from .plot import (EFFORT_MARKER, ERR_KW, GRID, INK, MUTED, PALETTE, X_LABEL, Judge, _cost_handles, _finish, _log_x,
                   _no_data, _panel_grid, _panels, _ref_handles, _references, _style, _ylim, effort_of, groups_of,
                   interval, tints, yerr)
from .runner import read_summary

log = logging.getLogger(__name__)

COPY_HATCH = "////"
MAX_DEPTH = 3  # how deep below a source to look for summary.csv
PART_RE = re.compile(r"part(\d+)")
SIZE_RE = re.compile(r"(.*[-_ /])(\d+(?:\.\d+)?)[bB]")  # a group ending in a parameter count, e.g. "GPT-oss-20b"
# Spread of the run variants within one group's hue: +-hue (of a full turn), and lightness from darker to lighter.
RUN_HUE_SPREAD, RUN_LIGHT_SPREAD = 0.06, 0.32


def _listing(src: Path) -> tuple[set[PurePosixPath], callable]:
    """(every file and folder path below src, relative, up to MAX_DEPTH levels; an opener (binary) for one of them)."""
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            names = {PurePosixPath(n.rstrip("/")) for n in z.namelist()}
        names |= {p for n in names for p in n.parents if p != PurePosixPath(".")}  # zips may omit folder entries

        @contextmanager
        def open_(rel):
            with zipfile.ZipFile(src) as z, z.open(str(rel)) as f:
                yield f
    else:
        names = {PurePosixPath(p.relative_to(src).as_posix()) for p in src.rglob("*")}

        def open_(rel):
            return (src / rel).open("rb")
    return {n for n in names if len(n.parts) <= MAX_DEPTH}, open_


def summary_paths(src: Path) -> list[PurePosixPath]:
    """The summary.csv files below src that make up its one run, merged in this order (later wins)."""
    names, _ = _listing(src)
    found = sorted((n for n in names if n.name == "summary.csv"), key=lambda n: (len(n.parts), str(n)))
    if PurePosixPath("summary.csv") in found:
        return [PurePosixPath("summary.csv")]
    # vllm_split.sh not merged yet: <split>/part<i>/; <split> may be src itself or one wrapping folder.
    splits = {n.parent.parent for n in found if PART_RE.fullmatch(n.parent.name)}
    if len(splits) == 1 and all(n.parent.parent == next(iter(splits)) for n in found):
        split = splits.pop()
        parts = sorted((n for n in names if n.parent == split and PART_RE.fullmatch(n.name)),
                       key=lambda n: int(PART_RE.fullmatch(n.name)[1]))
        if missing := [p.name for p in parts if p / "summary.csv" not in names]:
            log.warning("%s: %s have no summary.csv (failed, timed out or not downloaded); their biases are missing",
                        src.name, missing)
        return [p / "summary.csv" for p in parts if p / "summary.csv" in names]
    if len(found) == 1:
        return found
    if not found:
        raise FileNotFoundError(f"No summary.csv in {src} (searched {MAX_DEPTH} levels down); "
                                f"download the run's summary.csv, or its part*/ folders for an unmerged split job")
    raise FileNotFoundError(f"{src} holds several runs ({', '.join(map(str, found))}); pass the one to compare")


def read_source(src: Path) -> pd.DataFrame:
    """One source's summary (see the module doc), parts merged."""
    paths = summary_paths(src)
    _, open_ = _listing(src)
    log.info("%s: reading %s", src, ", ".join(map(str, paths)))
    frames = []
    for p in paths:
        with open_(p) as f:
            frames.append(read_summary(f))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["judge", "bias"], keep="last").reset_index(drop=True)


def load_runs(sources: list[Path]) -> pd.DataFrame:
    """All summaries in one frame. `model` = the group, suffixed with " [<run code>]" when two sources share a group
    (" [<run code>#<i>]" if the same run is the i-th source and another one too); `judge` = "<model> (<effort>)",
    unique across sources; `kind` = the group as the run named it."""
    frames = []
    for i, d in enumerate(sources):
        df = read_source(d)
        df["source"] = i
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    shared = df.groupby("group")["source"].transform("nunique") > 1
    if old := sorted({str(sources[i]) for i in df.loc[shared & df["code"].isna(), "source"]}):
        raise ValueError(f"{', '.join(old)}: summary.csv has no run code, which compare needs to tell apart the "
                         f"groups several runs share: {sorted(set(df.loc[shared, 'group']))}")
    reused = df.groupby(["group", "code"])["source"].transform("nunique") > 1  # the same run given twice
    df["kind"] = df["group"]
    df["model"] = [f"{g} [{c}#{s + 1}]" if r else f"{g} [{c}]" if sh else g
                   for g, c, s, sh, r in zip(df["group"], df["code"], df["source"], shared, reused)]
    df["judge"] = [m if effort_of(e) is None else f"{m} ({e})" for m, e in zip(df["model"], df["effort"])]
    return df.drop_duplicates(["judge", "bias"], keep="last").reset_index(drop=True)


def _name_key(group: str) -> tuple:
    """Groups by name, except that a last segment like "20b" sorts by its number: GPT-oss-20b before GPT-oss-120b."""
    m = SIZE_RE.fullmatch(group)
    return (m[1], float(m[2])) if m else (group, -1.0)


def _models(df: pd.DataFrame) -> dict[str, list[Judge]]:
    """{model: its judges low to high effort}, models sorted by group (see _name_key; models sharing one by their
    full name, i.e. by run code, so the order is fixed)."""
    models: dict[str, list[Judge]] = {}
    order = sorted(dict.fromkeys(zip(df["kind"], df["model"])), key=lambda km: (_name_key(km[0]), km[1]))
    for kind, model in order:
        sub = df[df["model"] == model]
        seen = list(dict.fromkeys(effort_of(e) for e in sub["effort"]))  # an empty cell reads back as NaN
        efforts = sorted(seen, key=lambda e: (effort_rank(e), seen.index(e)))
        models[model] = [Judge(model if e is None else f"{model} ({e})", model, model, e, kind=kind) for e in efforts]
    return models


def _variant(color: str, k: int, n: int) -> str:
    """The k-th of n variants of a hue (a group's runs): hue nudged and lightness stepped, darker to lighter."""
    if n == 1:
        return color
    h, l, s = rgb_to_hls(*to_rgb(color))
    x = k / (n - 1) - 0.5  # -0.5 .. 0.5
    return to_hex(hls_to_rgb((h + 2 * RUN_HUE_SPREAD * x) % 1, min(max(l + RUN_LIGHT_SPREAD * x, 0.2), 0.75), s))


def _colors(models: dict[str, list[Judge]]) -> dict[str, str]:
    """Each kind (a group as its run named it) gets one hue, its runs variants of it (_variant), and each run's efforts
    tints of that variant (lowest lightest), as in the single-run figures."""
    kinds: dict[str, list] = {}
    for members in models.values():
        kinds.setdefault(members[0].kind, []).append(members)
    colors = {}
    for i, runs in enumerate(kinds.values()):
        for r, members in enumerate(runs):
            colors.update(tints(members, _variant(PALETTE[i % len(PALETTE)], r, len(runs))))
    return colors


def tiers(models: dict[str, list[Judge]]) -> list[list[tuple[Judge, bool]]]:
    """tiers[t] = [(judge, is_repeat) per model]: bottom-aligned, a model's highest effort repeated upward."""
    n = max(len(m) for m in models.values())
    return [[(m[min(t, len(m) - 1)], t >= len(m)) for m in models.values()] for t in range(n)]


def _with_data(df: pd.DataFrame, cols: list[str], out: Path) -> pd.DataFrame:
    """Only models with some row where all `cols` are known (e.g. FLOPs exist for vLLM-served judges only; a summary
    from before a column existed has none)."""
    present = [c for c in cols if c in df]
    known = df[present].apply(pd.to_numeric, errors="coerce").notna().all(axis=1) if len(present) == len(cols) else \
        pd.Series(False, index=df.index)
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
    names = _panels(df, ycol)
    if not names:
        return None
    fig, panels = _panel_grid(names, ncols)
    with_refs = []
    for ax, panel in panels:
        sub = df[df["bias"] == panel.name].set_index("judge")
        lo, hi = interval(sub, ycol)
        pick = lambda s, j: float(s[j.label]) if j.label in sub.index else math.nan
        values, placed = [], []
        for t, tier in enumerate(layers):
            for k, (j, repeat) in enumerate(tier):
                v, x = pick(sub[ycol], j), t - 0.4 + width * (k + 0.5)
                values.append(v)
                placed.append((x, j))
                ax.bar(x, v, width=width, color=colors[j.label],
                       hatch=COPY_HATCH if repeat else None, edgecolor="white", linewidth=1, zorder=2,
                       yerr=yerr([v], [pick(lo, j)], [pick(hi, j)]), error_kw=ERR_KW)
        refs = _references(ax, sub, ycol, placed, width / 2)
        if refs:
            with_refs.append(panel)
        if ycol == "effect":
            ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
        ax.set_ylim(*_ylim(panel.scale(ycol), values + refs + [*lo, *hi]))
        _style(ax, panel.title, panel.ylabel(ycol))
        ax.set_xticks(range(len(layers)), [f"T{t + 1}" for t in range(len(layers))])
        ax.set_xlabel("Effort tier (low → high)", fontsize=8, color=MUTED)

    handles = [Patch(facecolor=colors[members[-1].label],
                     label=f"{model}: {' < '.join(j.effort or 'default' for j in members)}")
               for model, members in models.items()]
    handles.append(Patch(facecolor="#c3c2b7", hatch=COPY_HATCH, edgecolor="white", label="highest effort, repeated"))
    handles += _ref_handles(ycol, with_refs)
    return _finish(fig, handles, out, ncol=3)


def _pooled(df: pd.DataFrame, ycol: str, xcol: str) -> tuple[pd.DataFrame, list[Judge], dict[str, str]]:
    """Each group's runs pooled into one model "<group> [<its run codes>]" (just "<group>" without codes): per effort
    and bias, the mean and SD (across runs; NaN for one run) of cost (x) and ycol (y), from rows where both are known
    (the same run given twice counts once). Returns (that frame, a row per judge x bias; the judges, groups in name
    order and efforts low to high; their colors, one hue per group, efforts tints of it)."""
    df = df.assign(x=pd.to_numeric(df[xcol], errors="coerce"), y=pd.to_numeric(df[ycol], errors="coerce"),
                   eff=[effort_of(e) or "" for e in df["effort"]])
    df = df[(df["x"] > 0) & df["y"].notna()].drop_duplicates(["kind", "code", "eff", "bias"])
    codes = df.dropna(subset=["code"]).groupby("kind")["code"].agg(lambda c: sorted(set(c)))
    name = {k: f"{k} [{', '.join(codes[k])}]" if k in codes else k for k in df["kind"].unique()}
    pooled = (df.groupby(["kind", "eff", "bias"])
              .agg(x=("x", "mean"), y=("y", "mean"), x_sd=("x", "std"), y_sd=("y", "std")).reset_index())
    pooled["judge"] = [f"{name[k]} ({e})" if e else name[k] for k, e in zip(pooled["kind"], pooled["eff"])]
    judges, colors = [], {}
    for i, kind in enumerate(sorted(name, key=_name_key)):
        efforts = sorted(set(pooled.loc[pooled["kind"] == kind, "eff"]), key=lambda e: effort_rank(e or None))
        members = [Judge(f"{name[kind]} ({e})" if e else name[kind], name[kind], name[kind], e or None, kind=kind)
                   for e in efforts]
        judges += members
        colors.update(tints(members, PALETTE[i % len(PALETTE)]))
    return pooled, judges, colors


def _sd_scatter(ax, judges: list[Judge], colors: dict[str, str], df: pd.DataFrame) -> None:
    """As plot._cost_scatter, with ±1 SD bars both ways (df: _pooled's rows of one bias, indexed by judge)."""
    present = [j for j in judges if j.label in df.index]
    for members in groups_of(present).values():
        if len(members) >= 2:
            labels = [j.label for j in members]
            ax.plot(df.loc[labels, "x"], df.loc[labels, "y"], color=MUTED, linewidth=1.2, alpha=0.7, zorder=2)
    for j in present:
        r = df.loc[j.label]
        ax.errorbar(r["x"], r["y"], xerr=0 if math.isnan(r["x_sd"]) else r["x_sd"],
                    yerr=0 if math.isnan(r["y_sd"]) else r["y_sd"], color=colors[j.label], elinewidth=0.8,
                    capsize=2, linestyle="none", alpha=0.6, zorder=2)
        ax.plot(r["x"], r["y"], marker=EFFORT_MARKER.get(j.effort, "o"), markersize=7, color=colors[j.label],
                markeredgecolor="white", markeredgewidth=1.5, linestyle="none", zorder=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)


def _cost(df: pd.DataFrame, out: Path, ycol: str, ncols: int, xcol: str = "out_tokens") -> Path | None:
    """xcol = "out_tokens" (mean output tokens / call) or "flops" (mean FLOPs / call, vLLM-served judges only); a
    group's runs pooled into one model (_pooled)."""
    df = _with_data(df, [ycol, xcol], out)
    if df.empty:
        return None
    pooled, judges, colors = _pooled(df, ycol, xcol)
    fig, panels = _panel_grid(_panels(df), ncols)
    for ax, panel in panels:
        sub = pooled[pooled["bias"] == panel.name].set_index("judge")
        _style(ax, panel.title, panel.ylabel(ycol))
        ax.set_xlabel(X_LABEL[xcol], fontsize=8, color=MUTED)
        if sub.empty:
            _no_data(ax, "token" if xcol == "out_tokens" else "FLOP")
            continue
        _sd_scatter(ax, judges, colors, sub)
        x_sd, y_sd = sub["x_sd"].fillna(0), sub["y_sd"].fillna(0)
        _log_x(ax, [v for v in [*sub["x"], *(sub["x"] + x_sd), *(sub["x"] - x_sd)] if v > 0])
        ax.set_ylim(*_ylim(panel.scale(ycol), [*sub["y"], *(sub["y"] + y_sd), *(sub["y"] - y_sd)]))
    handles = _cost_handles(judges, colors)
    handles[-1].set_label("Same model, effort ↑ (bars: ±1 SD across its runs)")
    return _finish(fig, handles, out, ncol=3)


def compare_all(sources: list[Path], out_dir: Path, ncols: int = 4) -> list[Path]:
    """The comparison figures for these sources (folders or zips, see the module doc), plus the combined summary they
    were drawn from: the three bar grids, the two cost grids, and the two cost grids with FLOPs as the cost (judges
    served by our vLLM job). A figure no model has data for is skipped."""
    df = load_runs(sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    outs = [_bar_grid(df, out_dir / "bias_grid.png", "value", ncols),
            _bar_grid(df, out_dir / "effect_grid.png", "effect", ncols),
            _bar_grid(df, out_dir / "correctness_grid.png", "correctness", ncols),
            _cost(df, out_dir / "cost_vs_performance.png", "value", ncols),
            _cost(df, out_dir / "cost_vs_correctness.png", "correctness", ncols),
            _cost(df, out_dir / "cost_vs_performance_flops.png", "value", ncols, "flops"),
            _cost(df, out_dir / "cost_vs_correctness_flops.png", "correctness", ncols, "flops")]
    return [p for p in outs if p is not None]
