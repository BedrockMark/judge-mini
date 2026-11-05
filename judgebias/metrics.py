"""Paper metrics as pure functions. `None` means an unparseable/failed judgement."""
from __future__ import annotations

import math
from collections.abc import Sequence


def robustness_rate(pairs: Sequence[tuple[str | None, str | None]], expect_change: bool = False) -> float:
    """Fraction of (original, perturbed) choices that agree; invalid counts as not robust.
    With expect_change (Fallacy-Oversight, paper App. C) a pair is robust when the choices differ instead."""
    if not pairs:
        return math.nan
    return sum(o is not None and p is not None and (o != p) == expect_change for o, p in pairs) / len(pairs)


def consistency_rate(repeats: Sequence[Sequence[str | None]]) -> float:
    """Fraction of items whose repeated original judgements are all identical and valid."""
    if not repeats:
        return math.nan
    return sum(r[0] is not None and all(x == r[0] for x in r) for r in repeats) / len(repeats)


def accuracy(choices: Sequence[str | None], correct: str) -> float:
    if not choices:
        return math.nan
    return sum(c == correct for c in choices) / len(choices)


def error_rate(pairs: Sequence[tuple[float | None, float | None]]) -> float:
    """Mean |observed - reference| / reference over valid pairs."""
    valid = [(o, r) for o, r in pairs if o is not None and r]
    if not valid:
        return math.nan
    return sum(abs(o - r) / r for o, r in valid) / len(valid)
