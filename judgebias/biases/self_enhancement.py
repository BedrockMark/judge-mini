"""Self-Enhancement: does a judge score its own answer differently from how the other judges score it?

error = |self_score - mean(other judges' scores)| / mean(other judges' scores). Needs >= 2 judges.
An example counts as correct when the self score is within SELF_TOLERANCE of the others' mean.
"""
from __future__ import annotations

import asyncio
import math

from ..metrics import error_rate
from ..parse import parse_rating
from .base import Bias, BiasResult, register, sample

SELF_TOLERANCE = 0.5


class SelfEnhancementBias(Bias):
    name, title, metric = "self_enhancement", "Self-Enhancement", "error_rate"

    async def evaluate(self, judge, ctx) -> BiasResult:
        others = [j for j in ctx.judges if j.label != judge.label]
        rows = ctx.data("alignment")
        if not others:
            return BiasResult(value=math.nan, n=0, extra={"note": "needs >= 2 judges"})
        T = ctx.T

        async def run_item(row: dict) -> dict:
            answer = await ctx.ask(judge, row["question"])
            if not answer:
                return ({"idx": row["_idx"], "self": None, "others": []},
                        sample(row["_idx"], "invalid", row["question"], "an answer to the question", answer,
                               "the judge produced no answer to score"))
            prompt = T.self_enhancement_bias_prompt(row["question"], row["answer1"], answer)
            outs = await asyncio.gather(*(ctx.ask(j, prompt) for j in [judge, *others]))
            scores = [parse_rating(o) for o in outs]
            rec = {"idx": row["_idx"], "self": scores[0], "others": scores[1:]}
            valid = [s for s in rec["others"] if s is not None]
            ref = sum(valid) / len(valid) if valid else None
            listed = ", ".join(f"{j.label}: {'?' if s is None else f'{s:g}'}" for j, s in zip(others, rec["others"]))
            if ref is None:
                desired = f"unknown: no other judge gave a parseable score ({listed})"
            else:
                desired = f"Rating: [[{ref:.3g}]] ± {SELF_TOLERANCE:g} (mean score from the other judges: {listed})"
            status = ("invalid" if rec["self"] is None or ref is None
                      else "correct" if abs(rec["self"] - ref) <= SELF_TOLERANCE else "wrong")
            return rec, sample(row["_idx"], status, prompt, desired, outs[0], "the judged answer is the judge's own")

        done = await asyncio.gather(*(run_item(r) for r in rows))
        records = [rec for rec, _ in done]
        pairs = []
        for rec in records:
            valid = [s for s in rec["others"] if s is not None]
            pairs.append((rec["self"], sum(valid) / len(valid) if valid else None))
        return BiasResult(
            value=error_rate(pairs),
            n=len(records),
            n_invalid=sum(s is None or o is None for s, o in pairs),
            records=records,
            samples=[s for _, s in done],
        )


register(SelfEnhancementBias())
