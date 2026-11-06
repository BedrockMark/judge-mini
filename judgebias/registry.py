"""Short run codes: every results folder a command creates (and every vllm_split.sh job) gets a 4-character code,
recorded in <results_dir>/runs.tsv, so `get`, `runs` and `plot --run` can name a run by its code instead of a path."""
from __future__ import annotations

import csv
import os
import secrets
import zipfile
from datetime import datetime
from pathlib import Path

ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o, 1/l/i: easy to read off a log and type
COLUMNS = ["code", "created", "kind", "folder", "note"]


def registry_path(results_dir: Path) -> Path:
    return results_dir / "runs.tsv"


def load(results_dir: Path) -> list[dict]:
    path = registry_path(results_dir)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def register(results_dir: Path, folder: Path, kind: str, note: str = "") -> str:
    """Append a new entry and return its code. folder is stored relative to results_dir when it lies inside it."""
    taken = {r["code"] for r in load(results_dir)}
    code = next(c for c in iter(lambda: "".join(secrets.choice(ALPHABET) for _ in range(4)), None) if c not in taken)
    try:
        folder = folder.resolve().relative_to(results_dir.resolve())
    except ValueError:
        folder = folder.resolve()
    path = registry_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = [code, f"{datetime.now():%Y-%m-%d %H:%M}", kind, folder.as_posix(), note.replace("\t", " ").replace("\n", " ")]
    header = "" if path.exists() else "\t".join(COLUMNS) + "\n"
    # One short O_APPEND write per entry, so jobs registering at the same time don't interleave lines.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (header + "\t".join(row) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return code


def lookup(results_dir: Path, code: str) -> dict | None:
    return next((r for r in load(results_dir) if r["code"] == code.strip().lower()), None)


def result_dir(results_dir: Path, entry: dict) -> Path:
    """The folder holding the finished result: a split job's merged/, otherwise the registered folder itself."""
    folder = Path(entry["folder"])
    folder = folder if folder.is_absolute() else results_dir / folder
    return folder / "merged" if entry["kind"] == "split" else folder


def export(results_dir: Path, code: str, dest: Path | None = None) -> Path:
    """Zip only the files a run's result needs (summary, figures, examples.zip, config.yaml, logs) into one archive:
    no subfolders (e.g. an older run's unzipped raw/ or examples/), no split parts, no cache."""
    entry = lookup(results_dir, code)
    if entry is None:
        raise FileNotFoundError(f"No run with code '{code}' in {registry_path(results_dir)}")
    folder = result_dir(results_dir, entry)
    if not (folder / "summary.csv").exists():
        detail = ""
        if entry["kind"] == "split":
            parts = sorted(folder.parent.glob("part*"))
            finished = ", ".join(f"{p.name}={'yes' if (p / 'summary.csv').exists() else 'no'}" for p in parts)
            detail = (f" (not merged yet; parts finished: {finished or 'none'}; "
                      f"a part that timed out or failed: bash vllm_split.sh --resume {entry['code']})")
        raise FileNotFoundError(f"Run '{code}' has no result in {folder}{detail}")
    dest = dest or results_dir / "exports" / f"{entry['code']}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.iterdir()):
            if f.is_file():
                z.write(f, f"{entry['code']}/{f.name}")
    return dest
