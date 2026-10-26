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
    n: int
    n_invalid: int = 0
    cr: float = math.nan  # consistency rate of the unperturbed prompt (robustness biases only)
    extra: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)  # per-item details for results/raw


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
