"""Self-Enhancement: does a judge score its own answer differently from how the other judges score it?

error = |self_score - mean(other judges' scores)| / mean(other judges' scores). Needs >= 2 judges.
An example counts as correct when the self score is within SELF_TOLERANCE of the others' mean.

The paper's "others" are other models. A run that holds one model at several efforts (a vllm_split.sh job) can only
compare against the same model's other efforts, which measures effort-to-effort rating noise rather than
self-preference; such rows are flagged (`extra.same_model_only`) and a warning is logged.
"""
from __future__ import annotations

import asyncio
import logging
import math

from ..metrics import error_rates, mean_ci
from ..parse import parse_rating
from .base import Bias, BiasResult, register, sample

log = logging.getLogger(__name__)
SELF_TOLERANCE = 0.5


class SelfEnhancementBias(Bias):
    name, title, metric = "self_enhancement", "Self-Enhancement", "error_rate"

    def warnings(self, judges) -> list[str]:
        if len(judges) < 2:
            return ["self_enhancement needs >= 2 judges; its panel will be empty"]
        if len({j.model for j in judges}) == 1:
            return ["self_enhancement compares a judge with the other judges, and all of them are the same model "
                    "(other efforts); its error rate will measure effort-to-effort rating noise, not self-preference"]
        return []

    async def evaluate(self, judge, ctx) -> BiasResult:
        others = [j for j in ctx.judges if j.label != judge.label]
        rows = ctx.data("alignment")
        if not others:
            return BiasResult(value=math.nan, n=0, extra={"note": "needs >= 2 judges"})
        same_model = all(j.model == judge.model for j in others)
        if same_model:
            log.warning("%s / self_enhancement: every other judge is the same model (%s) at another effort; the "
                        "error rate measures effort-to-effort rating noise, not self-preference", judge.label, judge.model)
        T = ctx.T

        async def run_item(row: dict) -> tuple[tuple, dict]:
            answer = await ctx.ask(judge, row["question"])
            if not answer:
                return ((None, None),
                        sample(row["_idx"], "invalid", row["question"], "an answer to the question", answer,
                               "the judge produced no answer to score"))
            prompt = T.self_enhancement_bias_prompt(row["question"], row["answer1"], answer)
            outs = await asyncio.gather(*(ctx.ask(j, prompt) for j in [judge, *others]))
            scores = [parse_rating(o) for o in outs]
            self_score, other_scores = scores[0], scores[1:]
            valid = [s for s in other_scores if s is not None]
            ref = sum(valid) / len(valid) if valid else None
            listed = ", ".join(f"{j.label}: {'?' if s is None else f'{s:g}'}" for j, s in zip(others, other_scores))
            if ref is None:
                desired = f"unknown: no other judge gave a parseable score ({listed})"
            else:
                desired = f"Rating: [[{ref:.3g}]] ± {SELF_TOLERANCE:g} (mean score from the other judges: {listed})"
            status = ("invalid" if self_score is None or ref is None
                      else "correct" if abs(self_score - ref) <= SELF_TOLERANCE else "wrong")
            return (self_score, ref), sample(row["_idx"], status, prompt, desired, outs[0],
                                            "the judged answer is the judge's own")

        done = await asyncio.gather(*(run_item(r) for r in rows))
        pairs = [pair for pair, _ in done]
        value, ci = mean_ci(error_rates(pairs))
        return BiasResult(
            value=value, ci=ci, n=len(pairs),
            n_invalid=sum(s is None or o is None for s, o in pairs),
            extra={"others": [j.label for j in others], "same_model_only": same_model},
            samples=[s for _, s in done],
        )


register(SelfEnhancementBias())
