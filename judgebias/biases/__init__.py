# Import order defines registration order, which is the panel order in the figure. A new built-in task module goes
# here; tasks outside the package come from config `plugins` / `tasks` (custom.py).
from . import pairwise, cot, self_enhancement, refinement_aware, sycophancy  # noqa: F401
from .base import BIASES, METRICS, Bias, BiasResult, Metric, metric_spec, register, sample
from .custom import load_custom

BUILTIN = list(BIASES)

__all__ = ["BIASES", "BUILTIN", "METRICS", "Bias", "BiasResult", "Metric", "load_custom", "metric_spec", "register",
           "sample"]
