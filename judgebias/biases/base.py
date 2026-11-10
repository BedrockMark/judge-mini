from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..config import JudgeSpec
    from ..context import Context

MetricKind = Literal["robustness", "accuracy", "error_rate"]


@dataclass
class BiasResult:
    value: float
    n: int  # dataset rows judged
    ci: tuple[float, float] = (math.nan, math.nan)  # 95% interval of value (see metrics.py)
    n_invalid: int = 0  # verdicts / scores that could not be parsed (or calls that failed)
    # Robustness biases only (NaN otherwise), see biases/pairwise.py:
    cr: float = math.nan  # consistency rate of the unperturbed prompt (metrics.consistency_rate)
    expected_rr: float = math.nan  # the RR a perturbation with no effect would get (metrics.expected_rr)
    effect: float = math.nan  # value - expected_rr: the perturbation's effect beyond the judge's own inconsistency
    effect_ci: tuple[float, float] = (math.nan, math.nan)  # 95% bootstrap interval of effect
    drift: float = math.nan  # net move of verdicts from answer 1 to answer 2 (metrics.drift)
    drift_ci: tuple[float, float] = (math.nan, math.nan)  # 95% bootstrap interval of drift
    base_correctness: float = math.nan  # share of unperturbed judgements picking the ground truth (CoT: plain prompt)
    extra: dict = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)  # per-call examples for results/examples, see sample()


def sample(idx, status: Literal["correct", "wrong", "invalid"], input: str, desired: str, output: str | None,
           note: str = "", extra: dict[str, str | None] | None = None) -> dict:
    """One judged call: the prompt sent, what an unbiased judge should answer, and what it answered. The thinking
    text of `output` and of each `extra` output (when they are Completions carrying one) is kept alongside."""
    shown: dict[str, str | None] = {}
    for title, text in (extra or {}).items():
        if getattr(text, "reasoning", None):
            shown[f"{title}: thinking"] = text.reasoning
        shown[title] = text
    return {"idx": idx, "status": status, "input": input, "desired": desired, "output": output,
            "thinking": getattr(output, "reasoning", None), "note": note, "extra": shown}


class Bias(ABC):
    name: str
    title: str  # panel title in the figure
    metric: MetricKind

    @abstractmethod
    async def evaluate(self, judge: "JudgeSpec", ctx: "Context") -> BiasResult: ...


BIASES: dict[str, Bias] = {}


def register(bias: Bias) -> Bias:
    if bias.name in BIASES:
        raise ValueError(f"Bias '{bias.name}' registered twice")
    BIASES[bias.name] = bias
    return bias
