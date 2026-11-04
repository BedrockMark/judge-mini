"""CoT: judge accuracy with the CoT prompt on the fact-related set (answer1 is the better answer)."""
from __future__ import annotations

import asyncio

from ..metrics import accuracy
from ..parse import parse_verdict
from .base import Bias, BiasResult, register, sample
from .pairwise import base_prompt, to_answer


class CoTBias(Bias):
    name, title, metric = "cot", "CoT", "accuracy"

    async def evaluate(self, judge, ctx) -> BiasResult:
        T, rows = ctx.T, ctx.data("fact_related")
        cot_prompts = [T.CotPrompt(r["question"], r["answer1"], r["answer2"]) for r in rows]
        plain_prompts = [base_prompt(r, T) for r in rows]
        outs = await asyncio.gather(*(ctx.ask(judge, p) for p in cot_prompts + plain_prompts))
        cot = [to_answer(parse_verdict(o), False) for o in outs[: len(rows)]]
        plain = [to_answer(parse_verdict(o), False) for o in outs[len(rows):]]
        return BiasResult(
            value=accuracy(cot, "1"),
            n=len(rows),
            n_invalid=sum(c is None for c in cot),
            extra={"no_cot_accuracy": accuracy(plain, "1")},
            records=[{"idx": r["_idx"], "cot": c, "plain": p} for r, c, p in zip(rows, cot, plain)],
            samples=[sample(r["_idx"], "invalid" if c is None else "correct" if c == "1" else "wrong", prompt,
                            "[[A]] (answer 1 is the factually correct one)", out)
                     for r, c, prompt, out in zip(rows, cot, cot_prompts, outs)],
        )


register(CoTBias())
