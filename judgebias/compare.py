"""Compare finished runs without calling any model: several downloaded results' summary.csv -> six figures.

A source is a folder or a zip (e.g. from `python -m judgebias get`) holding one run's summary.csv: at its top, in a
single wrapping subfolder, or a vllm_split.sh folder's merged/ and/or part<i>/ summaries (merged in that order, a later
one winning on the same judge x bias, so a part rerun after the merge counts and a missing merge is not needed).

Each (folder, group) is one model; a group in several folders is told apart by the run code their summary.csv records
(older summaries without one: the code runs.tsv has for the folder, see registered_code). In the two bar grids, bars are clustered by effort tier: a model's efforts, sorted
low to high, fill tiers 1, 2, ... from the bottom, and a model with fewer efforts repeats its highest one in the upper
tiers (hatched bars). E.g. A {none, high} and B {low, medium, high} give tiers [A none, B low], [A high, B medium],
[A high (repeat), B high]. The cost grids (output tokens, and FLOPs for vLLM-served judges) overlay every model's judges in one scatter per bias."""
from __future__ import annotations

import io
import logging
import math
import re
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
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
MAX_DEPTH = 3  # how deep below a source to look for summary.csv
PART_RE = re.compile(r"part(\d+)")
# The output folder a copied run / plot log names (the folder's name on the server, before it was downloaded / renamed).
LOG_FOLDER_RE = re.compile(r"(?:output folder: |plotting .* -> )(.+?)\s*$")


def _effort(v) -> str | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or v == "" else str(v)


def _group(row) -> str:
    """Summaries written before the `group` column: the judge label without its " (<effort>)" suffix."""
    g = row.get("group")
    if isinstance(g, str) and g:
        return g
    m = re.fullmatch(r"(.*) \((.+)\)", str(row["judge"]))
    return m.group(1) if m and m.group(2) == row["effort"] else str(row["judge"])


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
        names = {PurePosixPath(p.relative_to(src).as_posix()) for p in src.rglob("*")
                 if len(p.relative_to(src).parts) <= MAX_DEPTH}

        def open_(rel):
            return (src / rel).open("rb")
    return {n for n in names if len(n.parts) <= MAX_DEPTH}, open_


def summary_paths(src: Path) -> list[PurePosixPath]:
    """The summary.csv files below src that make up its one run, merged in this order (later wins)."""
    names, _ = _listing(src)
    found = sorted((n for n in names if n.name == "summary.csv"), key=lambda n: (len(n.parts), str(n)))
    if PurePosixPath("summary.csv") in found:
        return [PurePosixPath("summary.csv")]
    # vllm_split.sh: <split>/merged/ and <split>/part<i>/; <split> may be src itself or one wrapping folder.
    splits = {n.parent.parent for n in found if n.parent.name == "merged" or PART_RE.fullmatch(n.parent.name)}
    if len(splits) == 1 and all(n.parent.parent == next(iter(splits)) for n in found):
        split = splits.pop()
        parts = sorted((n for n in names if n.parent == split and PART_RE.fullmatch(n.name)),
                       key=lambda n: int(PART_RE.fullmatch(n.name)[1]))
        if missing := [p.name for p in parts if p / "summary.csv" not in names]:
            log.warning("%s: %s have no summary.csv (failed, timed out or not downloaded); their biases are missing "
                        "unless the merged result has them", src.name, missing)
        chosen = [split / "merged" / "summary.csv"] + [p / "summary.csv" for p in parts]
        return [n for n in chosen if n in names]
    if len(found) == 1:
        return found
    if not found:
        raise FileNotFoundError(f"No summary.csv in {src} (searched {MAX_DEPTH} levels down); "
                                f"download the run's summary.csv, or its merged/ or part*/ folders for a split job")
    raise FileNotFoundError(f"{src} holds several runs ({', '.join(map(str, found))}); pass the one to compare")


def read_source(src: Path) -> pd.DataFrame:
    """One source's summary (see the module doc), parts merged."""
    paths = summary_paths(src)
    _, open_ = _listing(src)
    log.info("%s: reading %s", src, ", ".join(map(str, paths)))
    frames = []
    for p in paths:
        with open_(p) as f:
            frames.append(pd.read_csv(f, dtype={"code": str}))  # a code like "2e45" must stay text
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["judge", "bias"], keep="last").reset_index(drop=True)


def _run_folder_name(path: PurePosixPath) -> str:
    """The name runs.tsv knows a result folder by: a split's merged/ or part<i>/ go by their split folder's."""
    parts = [x for x in path.parts if x not in (".", "", "/")]
    if parts and (parts[-1] == "merged" or PART_RE.fullmatch(parts[-1])):
        parts.pop()
    return parts[-1] if parts else ""


def registered_code(src: Path, entries: list[dict]) -> str | None:
    """The run code runs.tsv (its entries) has for src, for summaries written before they recorded it. Names tried in
    order: src's own folder name (a split's merged/ or part<i>/ by its split folder's; a zip by its wrapping folder or
    its own name, e.g. `get`'s <code>/), then the server folder named in the run / plot log copied next to each
    summary, so a downloaded folder may have been renamed. A name matches an entry's folder name, or else its code."""
    names, open_ = _listing(src)
    paths = summary_paths(src)
    top = PurePosixPath(src.stem) if src.is_file() else PurePosixPath(src.resolve().as_posix())
    candidates = [_run_folder_name(top / p.parent) for p in paths] + [_run_folder_name(top)]
    for p in paths:
        for log_file in sorted(n for n in names if n.parent == p.parent and n.suffix == ".log"):
            with open_(log_file) as f:
                for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                    if m := LOG_FOLDER_RE.search(line):
                        candidates.append(_run_folder_name(PurePosixPath(m[1].replace("\\", "/"))))
    by_folder = {PurePosixPath(e["folder"]).name: e["code"] for e in entries}
    codes = {e["code"] for e in entries}
    for name in dict.fromkeys(c for c in candidates if c):
        if code := by_folder.get(name) or (name.lower() if name.lower() in codes else None):
            log.info("%s: summary.csv has no run code; runs.tsv has %s for folder %s", src, code, name)
            return code
    return None


def load_runs(sources: list[Path], entries: list[dict] | None = None) -> pd.DataFrame:
    """All summaries in one frame. `model` = the group, suffixed with " [<run code>]" when two sources share a group
    (" [<run code>#<i>]" if the same run is the i-th source and another one too); `judge` = "<model> (<effort>)",
    unique across sources. Rows without a run code (summaries written before they were recorded) get the one runs.tsv
    has for their source (entries = registry.load(), see registered_code); a shared group still without one is an
    error."""
    frames = []
    for i, d in enumerate(sources):
        df = _with_cost_columns(read_source(d))
        if "flops" not in df.columns:  # summaries written before FLOPs, or without any vLLM judge
            df["flops"] = math.nan
        if "code" not in df.columns:
            df["code"] = None
        if df["code"].isna().any() and entries and (code := registered_code(d, entries)):
            df["code"] = df["code"].fillna(code)
        df["effort"] = df["effort"].map(_effort).astype(object)
        df["group"] = df.apply(_group, axis=1)
        df["source"] = i
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    shared = df.groupby("group")["source"].transform("nunique") > 1
    if old := sorted({str(sources[i]) for i in df.loc[shared & df["code"].isna(), "source"]}):
        raise ValueError(f"Not supported by compare: {', '.join(old)} {'has' if len(old) == 1 else 'have'} no run code "
                         f"in summary.csv (written before run codes were recorded there) and none matched in runs.tsv "
                         f"(by folder name, or the server folder its copied log names); a code is needed to tell apart "
                         f"the groups several runs share: {sorted(set(df.loc[shared, 'group']))}")
    reused = df.groupby(["group", "code"])["source"].transform("nunique") > 1  # the same run given twice
    df["model"] = [f"{g} [{c}#{s + 1}]" if r else f"{g} [{c}]" if sh else g
                   for g, c, s, sh, r in zip(df["group"], df["code"], df["source"], shared, reused)]
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


def compare_all(sources: list[Path], out_dir: Path, ncols: int = 4, entries: list[dict] | None = None) -> list[Path]:
    """The comparison figures for these sources (folders or zips, see the module doc), plus the combined summary they were drawn from: the two bar
    grids, the two cost grids, and the two cost grids with FLOPs as the cost (judges served by our vLLM job). A
    figure no model has data for is skipped."""
    df = load_runs(sources, entries)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    outs = [_bar_grid(df, out_dir / "bias_grid.png", "value", ncols),
            _bar_grid(df, out_dir / "correctness_grid.png", "correctness", ncols),
            _cost_grid(df, out_dir / "cost_vs_performance.png", "value", ncols),
            _cost_grid(df, out_dir / "cost_vs_correctness.png", "correctness", ncols),
            _cost_grid(df, out_dir / "cost_vs_performance_flops.png", "value", ncols, "flops"),
            _cost_grid(df, out_dir / "cost_vs_correctness_flops.png", "correctness", ncols, "flops")]
    return [p for p in outs if p is not None]
