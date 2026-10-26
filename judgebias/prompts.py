"""Loads the paper's promptTemplate.py from the dataset folder (not copied) and adds
the one template it lacks: refinement-aware scoring."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=None)
def load_templates(dataset_dir: Path) -> ModuleType:
    path = Path(dataset_dir) / "promptTemplate.py"
    spec = importlib.util.spec_from_file_location("promptTemplate", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refinement_aware_score(T: ModuleType, question: str, original: str, refined: str) -> str:
    """get_score prompt where the judge is told the response was refined from an earlier answer."""
    answer = (
        "Note: the response below is a refined version of an earlier answer.\n"
        f"[Earlier Answer]\n{original}\n[Refined Answer]\n{refined}"
    )
    return T.get_score(question, answer)
