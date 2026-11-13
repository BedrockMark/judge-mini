"""CoT: judge accuracy with the CoT prompt on the fact-related set (answer1 is the better answer). The same rows are
also judged with the plain prompt: its accuracy is `expected`, and the gain the CoT prompt brings is `effect` (with a
bootstrap interval over rows, the two prompts resampled together)."""
from __future__ import annotations

import asyncio

from ..metrics import bootstrap_ci, proportion
from ..parse import parse_verdict
from .base import Bias, BiasResult, register, sample
from .pairwise import base_prompt, to_answer


def _status(verdict: str | None) -> str:
    return "invalid" if verdict is None else "correct" if verdict == "1" else "wrong"


def _gain(pairs: list[tuple[str | None, str | None]]) -> float:
    """CoT accuracy - plain accuracy over these (cot, plain) verdict pairs."""
    return proportion([c == "1" for c, _ in pairs])[0] - proportion([p == "1" for _, p in pairs])[0]


class CoTBias(Bias):
    name, title, metric = "cot", "CoT", "accuracy"
    case_note = ("Bias better case: the plain (no-CoT) prompt's verdict was wrong but the CoT prompt's was correct; "
                 "bias worse case: the reverse. Answer 1 is always the correct pick.")

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
            expected=plain_acc, effect=value - plain_acc, effect_ci=bootstrap_ci(list(zip(cot, plain)), _gain),
            base_correctness=plain_acc,
            samples=[sample(r["_idx"], _status(c), prompt, "[[A]] (answer 1 is the factually correct one)", out,
                            extra={"Output to the plain (no-CoT) prompt": plain_out}, base_status=_status(p))
                     for r, c, p, prompt, out, plain_out in zip(rows, cot, plain, cot_prompts, outs, outs[len(rows):])],
        )


register(CoTBias())
