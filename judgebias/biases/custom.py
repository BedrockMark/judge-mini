"""Tasks defined outside the built-in panels, loaded from the config:

- `plugins`: Python files (or importable module names) that define and register() their own Bias, e.g. a
  PerturbationBias with a `variants` function, or a Bias subclass with a metric of its own.
- `tasks`: declarative pairwise perturbations needing no code: each variant swaps the row's question / answer1 /
  answer2 for other fields of the same row (and may show the answers swapped). Such a task is a PerturbationBias
  measured by Robustness Rate like the built-in ones; rows lacking a variant's field skip that variant.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BIASES, register
from .pairwise import PerturbationBias, Variant, VariantFn

if TYPE_CHECKING:
    from ..config import Config, TaskSpec, VariantSpec

FIELDS = ("question", "answer1", "answer2")
_PLUGINS: dict[str, list[str]] = {}  # plugin reference -> the biases its import registered


class DeclaredBias(PerturbationBias):
    """A PerturbationBias built from a config `tasks` entry (re-registered whenever a config declares it again)."""


def field_variants(specs: list["VariantSpec"]) -> VariantFn:
    def variants(row, T, opts):
        out = []
        for v in specs:
            swap = {f: getattr(v, f) for f in FIELDS if getattr(v, f)}
            if not all(row.get(src) for src in swap.values()):
                continue  # skipped, not invented
            q, a1, a2 = (row[swap.get(f, f)] for f in FIELDS)
            out.append(Variant(v.name, T.evaluate_ai_responses_no_tie(q, *((a2, a1) if v.swapped else (a1, a2))),
                               swapped=v.swapped))
        return out
    return variants


def declare(spec: "TaskSpec") -> DeclaredBias:
    from ..config import ConfigError
    if spec.name in BIASES and not isinstance(BIASES[spec.name], DeclaredBias):
        raise ConfigError(f"Task '{spec.name}' in `tasks` clashes with a built-in or plugin bias of that name")
    return register(DeclaredBias(spec.name, spec.title or spec.name, spec.dataset, field_variants(spec.variants),
                                 spec.expect_change), replace=True)


def load_plugin(ref: str) -> list[str]:
    """Import a plugin once (a .py path, or a module name) and return the biases it registered."""
    from ..config import ConfigError
    if ref in _PLUGINS:
        return _PLUGINS[ref]
    before = set(BIASES)
    if ref.endswith(".py"):
        path = Path(ref)
        if not path.is_file():
            raise ConfigError(f"Plugin {path} not found")
        name = f"judgebias_plugin_{path.stem}_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        importlib.import_module(ref)
    _PLUGINS[ref] = [b for b in BIASES if b not in before]
    return _PLUGINS[ref]


def load_custom(cfg: "Config") -> list[str]:
    """Register the config's plugins and declared tasks; returns their bias names in config order."""
    names = [b for ref in cfg.plugins for b in load_plugin(ref)]
    names += [declare(t).name for t in cfg.tasks]
    return list(dict.fromkeys(names))
