"""Refinement-Aware: does telling the judge an answer was refined change its score?

error = |aware_score - unaware_score| / unaware_score.
"""
from __future__ import annotations

import asyncio

from ..metrics import error_rates, mean_ci
from ..parse import parse_rating
from ..prompts import refinement_aware_score
from .base import Bias, BiasResult, register, sample


class RefinementAwareBias(Bias):
    name, title, metric = "refinement_aware", "Refinement-Aware", "error_rate"

    async def evaluate(self, judge, ctx) -> BiasResult:
        T, rows = ctx.T, ctx.data("refinement")

        async def run_item(row: dict) -> tuple[tuple, dict]:
            aware_prompt = refinement_aware_score(T, row["question"], row["answer1"], row["answer1_polished"])
            unaware, aware = await asyncio.gather(
                ctx.ask(judge, T.get_score(row["question"], row["answer1_polished"])),
                ctx.ask(judge, aware_prompt),
            )
            a, u = parse_rating(aware), parse_rating(unaware)
            if u is None:
                desired = "unknown: the score without the refinement note could not be parsed"
            else:
                desired = f"Rating: [[{u:g}]] (score given to the same answer without the refinement note)"
            status = "invalid" if a is None or u is None else "correct" if a == u else "wrong"
            return (a, u), sample(row["_idx"], status, aware_prompt, desired, aware,
                                  extra={"Output without the refinement note": unaware})

        done = await asyncio.gather(*(run_item(r) for r in rows))
        pairs = [pair for pair, _ in done]
        value, ci = mean_ci(error_rates(pairs))
        return BiasResult(
            value=value, ci=ci, n=len(pairs),
            n_invalid=sum(a is None or u is None for a, u in pairs),
            samples=[s for _, s in done],
        )


register(RefinementAwareBias())
