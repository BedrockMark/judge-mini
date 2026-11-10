"""Paper metrics as pure functions, each with a 95% confidence interval. `None` means an unparseable/failed judgement.

Proportions (RR, CR, accuracy, correctness) get a Wilson interval; error rates (a mean of per-item ratios) a normal
interval on the mean. Items are treated as independent; a bias with several perturbations per dataset row
(authority, sentiment, diversity) counts each perturbation as an item, so its interval is slightly too narrow.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

Z = 1.96  # 95%
Choice = str | None

NAN_CI = (math.nan, math.nan)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval of a proportion k / n."""
    if n == 0:
        return NAN_CI
    p, z2 = k / n, Z * Z
    centre, half = (p + z2 / (2 * n)) / (1 + z2 / n), Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(flags: Sequence[bool]) -> tuple[float, tuple[float, float]]:
    """(share of True, 95% CI); NaN when empty."""
    if not flags:
        return math.nan, NAN_CI
    return sum(flags) / len(flags), wilson(sum(flags), len(flags))


def mean_ci(values: Sequence[float]) -> tuple[float, tuple[float, float]]:
    """(mean, 95% normal interval of the mean); NaN when empty, a point interval for a single value."""
    if not values:
        return math.nan, NAN_CI
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, (m, m)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    half = Z * sd / math.sqrt(len(values))
    return m, (max(0.0, m - half), m + half)


def is_robust(original: Choice, perturbed: Choice, expect_change: bool = False) -> bool:
    """A (original, perturbed) pair is robust when both are valid and agree; with expect_change (Fallacy-Oversight,
    paper App. C: answer 1 is made chaotic, so the right verdict moves) when they differ instead."""
    return original is not None and perturbed is not None and (original != perturbed) == expect_change


def robustness_rate(pairs: Sequence[tuple[Choice, Choice]], expect_change: bool = False) -> float:
    return proportion([is_robust(o, p, expect_change) for o, p in pairs])[0]


def drift(pairs: Sequence[tuple[Choice, Choice]]) -> float:
    """Net share of items whose verdict moved from answer 1 to answer 2 under the perturbation, in -1..1: positive =
    toward answer 2, negative = toward answer 1, ~0 = the flips are symmetric (noise, not a directed bias). Pairs with
    an invalid side do not move."""
    if not pairs:
        return math.nan
    to_2 = sum(o == "1" and p == "2" for o, p in pairs)
    to_1 = sum(o == "2" and p == "1" for o, p in pairs)
    return (to_2 - to_1) / len(pairs)


def consistency_rate(repeats: Sequence[Sequence[Choice]]) -> float:
    """Share of items whose repeated original judgements are all identical and valid."""
    return proportion([r[0] is not None and all(x == r[0] for x in r) for r in repeats])[0]


def accuracy(choices: Sequence[Choice], correct: str) -> float:
    return proportion([c == correct for c in choices])[0]


def error_rates(pairs: Sequence[tuple[float | None, float | None]]) -> list[float]:
    """|observed - reference| / reference for every valid pair (reference > 0)."""
    return [abs(o - r) / r for o, r in pairs if o is not None and r]


def error_rate(pairs: Sequence[tuple[float | None, float | None]]) -> float:
    """Mean |observed - reference| / reference over valid pairs."""
    return mean_ci(error_rates(pairs))[0]
