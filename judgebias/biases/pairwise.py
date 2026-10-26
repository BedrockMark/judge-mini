"""Generic "original vs perturbed" pairwise bias measured by Robustness Rate, plus the 9 panels using it.

To add a bias: write a `variants(row, T, opts)` function returning Variant(s) and register a
PerturbationBias. Choices are normalised to which *underlying* answer won ("1" or "2"), so position
swaps are comparable to unswapped prompts.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from ..metrics import consistency_rate, robustness_rate
from ..parse import parse_verdict
from .base import Bias, BiasResult, register


@dataclass
class Variant:
    name: str
    prompt: str
    swapped: bool = False  # True when answer1 is shown in position B


VariantFn = Callable[[dict, ModuleType, dict], list[Variant]]


def to_answer(verdict: str | None, swapped: bool) -> str | None:
    if verdict not in ("A", "B"):
        return None
    first = verdict == "A"
    return "1" if first != swapped else "2"


def base_prompt(row: dict, T: ModuleType, a1: str | None = None, a2: str | None = None) -> str:
    return T.evaluate_ai_responses_no_tie(row["question"],
                                          row["answer1"] if a1 is None else a1,
                                          row["answer2"] if a2 is None else a2)


class PerturbationBias(Bias):
    metric = "robustness"

    def __init__(self, name: str, title: str, dataset: str, variants: VariantFn):
        self.name, self.title, self.dataset, self.variants = name, title, dataset, variants

    async def evaluate(self, judge, ctx) -> BiasResult:
        T, opts = ctx.T, ctx.options(self.name)

        async def run_item(row: dict) -> dict:
            orig_prompt = base_prompt(row, T)
            variants = self.variants(row, T, opts)
            outs = await asyncio.gather(
                *(ctx.ask(judge, orig_prompt, r) for r in range(ctx.cfg.cr_repeats)),
                *(ctx.ask(judge, v.prompt) for v in variants),
            )
            orig = [to_answer(parse_verdict(o), False) for o in outs[: ctx.cfg.cr_repeats]]
            pert = {v.name: to_answer(parse_verdict(o), v.swapped)
                    for v, o in zip(variants, outs[ctx.cfg.cr_repeats:])}
            return {"idx": row["_idx"], "original": orig, "perturbed": pert}

        records = await asyncio.gather(*(run_item(r) for r in ctx.data(self.dataset)))
        pairs = [(rec["original"][0], p) for rec in records for p in rec["perturbed"].values()]
        all_choices = [c for rec in records for c in [*rec["original"], *rec["perturbed"].values()]]
        return BiasResult(
            value=robustness_rate(pairs),
            cr=consistency_rate([rec["original"] for rec in records]),
            n=len(records),
            n_invalid=sum(c is None for c in all_choices),
            records=records,
        )


# --- the 9 robustness panels, in Goal.png order -------------------------------------------------

def _position(row, T, opts):
    return [Variant("swapped", base_prompt(row, T, row["answer2"], row["answer1"]), swapped=True)]


def _verbosity(row, T, opts):
    return [Variant("longer", base_prompt(row, T, a2=row["answer2_longer"]))]


def _compassion_fade(row, T, opts):
    a, b = opts.get("model_names", ["GPT-4o", "Claude-3.5-Sonnet"])
    return [Variant(f"{a}|{b}", T.compassion_fade_prompt(row["question"], row["answer1"], row["answer2"],
                                                         model_a=a, model_b=b))]


def _bandwagon(row, T, opts):
    # Social proof points at the rejected answer (answer2, shown as Assistant B).
    return [Variant("B", T.Bandwagon_effect_prompt(row["question"], row["answer1"], row["answer2"],
                                                   number=opts.get("number", 90), chosen_model="Assistant B"))]


def _distraction(row, T, opts):
    return [Variant("B", T.Dstraction_Bias_B(row["question"], row["answer1"], row["answer2"]))]


def _fallacy(row, T, opts):
    return [Variant("fallacy", base_prompt(row, T, a1=row["answer1_fallacy_oversight"]))]


# Some authority/sentiment rows lack a perturbed field; those variants are skipped, not invented.
def _authority(row, T, opts):
    return [Variant(k, base_prompt(row, T, a2=row[f"answer2_with_reference_{k}"]))
            for k in opts.get("kinds", ["book", "quote", "url"]) if row.get(f"answer2_with_reference_{k}")]


def _sentiment(row, T, opts):
    return [Variant(t, base_prompt(row, T, a1=row[f"answer1_{t}"]))
            for t in opts.get("tones", ["cheerful", "sad", "angry", "fear"]) if row.get(f"answer1_{t}")]


def _diversity(row, T, opts):
    return [Variant(g, T.Diversity_bias_prompt(row["question"], row["answer1"], row["answer2"],
                                               chosen_model="Assistant B", group=g))
            for g in opts.get("groups", ["Homosexual", "Black", "Female", "HIV Positive"])]


for _name, _title, _ds, _fn in [
    ("position", "Position", "alignment", _position),
    ("verbosity", "Verbosity", "verbosity", _verbosity),
    ("compassion_fade", "Compassion-Fade", "alignment", _compassion_fade),
    ("bandwagon", "Bandwagon", "alignment", _bandwagon),
    ("distraction", "Distraction", "alignment", _distraction),
    ("fallacy_oversight", "Fallacy-Oversight", "fallacy_oversight", _fallacy),
    ("authority", "Authority", "authority", _authority),
    ("sentiment", "Sentiment", "sentiment", _sentiment),
    ("diversity", "Diversity", "alignment", _diversity),
]:
    register(PerturbationBias(_name, _title, _ds, _fn))
