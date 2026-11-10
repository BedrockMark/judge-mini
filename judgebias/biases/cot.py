"""CoT: judge accuracy with the CoT prompt on the fact-related set (answer1 is the better answer). The same rows are
also judged with the plain prompt; `extra` keeps that accuracy and the gain the CoT prompt brings."""
from __future__ import annotations

import asyncio

from ..metrics import proportion
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
        value, ci = proportion([c == "1" for c in cot])
        plain_acc = proportion([c == "1" for c in plain])[0]
        return BiasResult(
            value=value, ci=ci, n=len(rows),
            n_invalid=sum(c is None for c in cot),
            base_correctness=plain_acc,
            extra={"no_cot_accuracy": plain_acc, "cot_gain": value - plain_acc},
            samples=[sample(r["_idx"], "invalid" if c is None else "correct" if c == "1" else "wrong", prompt,
                            "[[A]] (answer 1 is the factually correct one)", out)
                     for r, c, prompt, out in zip(rows, cot, cot_prompts, outs)],
        )


register(CoTBias())
