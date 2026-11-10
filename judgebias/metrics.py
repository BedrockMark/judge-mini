"""Paper metrics as pure functions, each with a 95% confidence interval. `None` means an unparseable/failed judgement.

Proportions (RR, CR, accuracy, correctness) get a Wilson interval; error rates (a mean of per-item ratios) a normal
interval on the mean. Items are treated as independent; a bias with several perturbations per dataset row
(authority, sentiment, diversity) counts each perturbation as an item, so its interval is slightly too narrow.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TypeVar

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


def majority(choices: Sequence[Choice]) -> Choice:
    """The verdict most of the valid repeats gave; a tie falls back to the first valid one (the paper's single
    judgement). None when no repeat is valid."""
    valid = [c for c in choices if c is not None]
    if not valid:
        return None
    top = Counter(valid).most_common()
    return valid[0] if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


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


def consistency_pairs(repeats: Sequence[Sequence[Choice]]) -> list[tuple[Choice, Choice]]:
    """Leave-one-out pairs (majority of the other repeats, held-out repeat), one per repeat of every item: each is
    an unperturbed judgement compared with the majority baseline, exactly what RR does with a perturbed one."""
    return [(majority([*r[:k], *r[k + 1:]]), r[k]) for r in repeats if len(r) > 1 for k in range(len(r))]


def consistency_rate(repeats: Sequence[Sequence[Choice]]) -> float:
    """Share of consistency_pairs that agree (both valid): how often a fresh unperturbed judgement matches the
    majority baseline. With 2 repeats this is the paper's CR (the two judgements agree). It is the RR a perturbation
    with no effect would get, so RR - CR is the perturbation's effect (see expected_rr for Fallacy-Oversight)."""
    return proportion([is_robust(o, p) for o, p in consistency_pairs(repeats)])[0]


def expected_rr(cr: float, expect_change: bool = False) -> float:
    """RR expected when the perturbation changes nothing: CR, or 1 - CR when the robust verdict is a *changed* one
    (Fallacy-Oversight), since then only inconsistency makes an item count as robust."""
    return 1 - cr if expect_change else cr


T = TypeVar("T")


def bootstrap_ci(items: Sequence[T], stat: Callable[[Sequence[T]], float], n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of stat(items), resampling items (dataset rows) with replacement; NaN
    resamples (e.g. a statistic with no valid pairs) are dropped."""
    if not items:
        return NAN_CI
    rng = random.Random(seed)
    values = sorted(v for _ in range(n_boot) if not math.isnan(v := stat(rng.choices(items, k=len(items)))))
    if not values:
        return NAN_CI
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def accuracy(choices: Sequence[Choice], correct: str) -> float:
    return proportion([c == correct for c in choices])[0]


def error_rates(pairs: Sequence[tuple[float | None, float | None]]) -> list[float]:
    """|observed - reference| / reference for every valid pair (reference > 0)."""
    return [abs(o - r) / r for o, r in pairs if o is not None and r]


def error_rate(pairs: Sequence[tuple[float | None, float | None]]) -> float:
    """Mean |observed - reference| / reference over valid pairs."""
    return mean_ci(error_rates(pairs))[0]
