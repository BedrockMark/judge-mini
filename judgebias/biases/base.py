"""What every bias test (task) shares: its result, its per-call samples, its metric, and the registry.

A task is a `Bias` subclass with `name`, `title`, `metric` and an async `evaluate(judge, ctx)` returning a
BiasResult; `register()` it and the runner, summary, examples, check and figures handle it with no other change.
Built-in tasks live in this package (imported in biases/__init__.py); others come from config `plugins` (Python files
or modules that call register()) or config `tasks` (declarative pairwise perturbations, biases/custom.py).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..config import JudgeSpec
    from ..context import Context


@dataclass(frozen=True)
class Metric:
    """How a panel's value reads in the figures. `bounded`: a 0-1 share (else the y range follows the data);
    `ref_label`: what BiasResult.expected is for this metric (the dashed mark over each bar); `effect_label`: the y
    label of value - expected in effect_grid.png."""
    name: str
    label: str
    bounded: bool = True
    ref_label: str = "Expected value if unbiased"
    effect_label: str = "Value − expected"


METRICS: dict[str, Metric] = {m.name: m for m in [
    Metric("robustness", "Robustness Rate", ref_label="Expected RR if unbiased (from CR)",
           effect_label="RR − expected RR"),
    Metric("accuracy", "Accuracy", ref_label="Accuracy on the unbiased prompt",
           effect_label="Accuracy − unbiased accuracy"),
    Metric("error_rate", "Error Rate (lower is better)", bounded=False),
]}


def metric_spec(name: str) -> Metric:
    """The registered Metric, or a default one (unbounded, titled from the name) for a metric only a summary knows."""
    return METRICS.get(name) or Metric(name, str(name).replace("_", " ").capitalize(), bounded=False)


@dataclass
class BiasResult:
    value: float
    n: int  # dataset rows judged
    ci: tuple[float, float] = (math.nan, math.nan)  # 95% interval of value (see metrics.py)
    n_invalid: int = 0  # verdicts / scores that could not be parsed (or calls that failed)
    # The value the judge would get if the bias had no effect (robustness: CR-based expected RR; CoT: the plain
    # prompt's accuracy), and value - expected with its 95% interval. NaN when the task has no such reference.
    expected: float = math.nan
    effect: float = math.nan
    effect_ci: tuple[float, float] = (math.nan, math.nan)
    base_correctness: float = math.nan  # share of judgements on the unbiased prompt that pick the ground truth
    stats: dict[str, float] = field(default_factory=dict)  # task-specific numbers, one summary column each
    extra: dict = field(default_factory=dict)  # anything else, stored as JSON in the summary's `extra`
    samples: list[dict] = field(default_factory=list)  # per-call examples for results/examples, see sample()


Status = Literal["correct", "wrong", "invalid"]


def sample(idx, status: Status, input: str, desired: str, output: str | None,
           note: str = "", extra: dict[str, str | None] | None = None, base_status: Status | None = None) -> dict:
    """One judged call: the prompt sent, what an unbiased judge should answer, and what it answered. The thinking
    text of `output` and of each `extra` output (when they are Completions carrying one) is kept alongside.
    `base_status` classifies the same row under the unbiased prompt, which examples.py pairs with `status` into the
    bias_better/bias_worse cases (and summary's base_correctness counts)."""
    shown: dict[str, str | None] = {}
    for title, text in (extra or {}).items():
        if getattr(text, "reasoning", None):
            shown[f"{title}: thinking"] = text.reasoning
        shown[title] = text
    return {"idx": idx, "status": status, "input": input, "desired": desired, "output": output, "base_status": base_status,
            "thinking": getattr(output, "reasoning", None), "note": note, "extra": shown}


class Bias(ABC):
    name: str
    title: str  # panel title in the figures
    metric: str  # a METRICS key (or any name: metric_spec gives it a default look)
    # examples.py: explains the bias_better/bias_worse files of a task whose samples carry a base_status
    case_note: str = ("Bias better case: the unbiased prompt's verdict was wrong but the biased prompt's was correct; "
                      "bias worse case: the reverse.")

    @abstractmethod
    async def evaluate(self, judge: "JudgeSpec", ctx: "Context") -> BiasResult: ...

    def warnings(self, judges: list["JudgeSpec"]) -> list[str]:
        """Setup problems `check` reports for these judges (e.g. a task that needs several models)."""
        return []


BIASES: dict[str, Bias] = {}


def register(bias: Bias, replace: bool = False) -> Bias:
    if bias.name in BIASES and not replace:
        raise ValueError(f"Bias '{bias.name}' registered twice")
    BIASES[bias.name] = bias
    return bias
