"""Dataset loading with a seed-fixed subset per dataset."""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

DATASET_FILES = {
    "alignment": "base_datasets/alignment_dataset.json",
    "fact_related": "base_datasets/fact_related_dataset.json",
    "refinement_aware": "base_datasets/refinement_aware_dataset.json",
    "authority": "bias_datasets/authority_bias_standardized.json",
    "fallacy_oversight": "bias_datasets/fallacy_oversight_bias_standardized.json",
    "refinement": "bias_datasets/refinement_bias_standardized.json",
    "sentiment": "bias_datasets/sentiment_bias_standardized.json",
    "verbosity": "bias_datasets/verbosity_bias_standardized.json",
}


@lru_cache(maxsize=None)
def _read(path: Path) -> tuple[dict, ...]:
    return tuple(json.loads(path.read_text(encoding="utf-8")))


def load_dataset(dataset_dir: Path, name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    """Return up to n rows (seeded sample, original order kept), each tagged with `_idx`."""
    rows = _read(Path(dataset_dir) / DATASET_FILES[name])
    idx = list(range(len(rows)))
    if n is not None and n < len(rows):
        idx = sorted(random.Random(seed).sample(idx, n))
    return [{**rows[i], "_idx": i} for i in idx]
