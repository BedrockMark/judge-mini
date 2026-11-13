"""Dataset loading with a seed-fixed subset per dataset. The built-in names map to the paper's files under
dataset_dir; config `datasets` adds (or replaces) names with files of its own: JSON (a list of row objects) or JSONL."""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

DATASET_FILES = {
    "alignment": "base_datasets/alignment_dataset.json",
    "fact_related": "base_datasets/fact_related_dataset.json",
    "authority": "bias_datasets/authority_bias_standardized.json",
    "fallacy_oversight": "bias_datasets/fallacy_oversight_bias_standardized.json",
    "refinement": "bias_datasets/refinement_bias_standardized.json",
    "sentiment": "bias_datasets/sentiment_bias_standardized.json",
    "verbosity": "bias_datasets/verbosity_bias_standardized.json",
}


@lru_cache(maxsize=None)
def _read(path: Path) -> tuple[dict, ...]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return tuple(json.loads(line) for line in text.splitlines() if line.strip())
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: a dataset must be a JSON list of rows (or a .jsonl file)")
    return tuple(rows)


def load_dataset(path: Path, n: int | None = None, seed: int = 0) -> list[dict]:
    """Return up to n rows (seeded sample, original order kept), each tagged with `_idx`."""
    rows = _read(Path(path))
    idx = list(range(len(rows)))
    if n is not None and n < len(rows):
        idx = sorted(random.Random(seed).sample(idx, n))
    return [{**rows[i], "_idx": i} for i in idx]
