"""Generic "original vs perturbed" pairwise bias measured by Robustness Rate, plus the 9 panels using it.

To add a bias: write a `variants(row, T, opts)` function returning Variant(s) and register a
PerturbationBias (in a plugin file, see config `plugins`; or, when the perturbation only swaps in other fields of the
row, declare it under config `tasks`, biases/custom.py). Rows need `question`, `answer1` (the better answer) and
`answer2`, unless `prompt` builds the unperturbed prompt differently. Choices are normalised to which *underlying*
answer won ("1" or "2"), so position swaps are comparable to unswapped prompts.

RR (paper Sec. 3): a perturbed verdict is robust when it matches the unperturbed one, except for
Fallacy-Oversight (paper App. C): answer 1 is made logically chaotic, so the robust verdict *changes*.
The unperturbed prompt is judged `cr_repeats` times; the baseline verdict is their majority (a tie: the first), which
is less noisy than the paper's single judgement. CR (metrics.consistency_rate) compares each repeat with the majority
of the others, so it is what RR would be if the perturbation did nothing, and `effect` = RR - expected RR (CR, or
1 - CR for Fallacy-Oversight) is the perturbation's own effect, with a bootstrap interval over dataset rows.
Correctness compares the perturbed verdict to the ground truth: answer 1, or answer 2 for Fallacy-Oversight;
base_correctness does the same for the unperturbed repeats (always answer 1).
Drift (metrics.drift) tells a directed bias from noise: the net share of verdicts that moved toward answer 2.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from ..metrics import bootstrap_ci, consistency_pairs, drift, expected_rr, is_robust, majority, proportion
from ..parse import parse_verdict
from .base import Bias, BiasResult, register, sample


@dataclass
class Variant:
    name: str
    prompt: str
    swapped: bool = False  # True when answer1 is shown in position B


VariantFn = Callable[[dict, ModuleType, dict], list[Variant]]
PromptFn = Callable[[dict, ModuleType], str]  # the unperturbed prompt of a row


@dataclass
class Item:
    """One dataset row's verdicts: the unperturbed repeats, (baseline, perturbed) pairs per variant, and the
    leave-one-out consistency pairs. The bootstrap resamples these."""
    orig: list[str | None]
    pairs: list[tuple[str | None, str | None]]
    cr_pairs: list[tuple[str | None, str | None]]


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
    case_note = ("Bias better case: the unbiased prompt's verdict was wrong (picked answer 2) but the biased prompt's "
                 "was correct; bias worse case: the reverse. On the unbiased prompt answer 1 is always correct; the "
                 "biased prompt is graded as in correct.md (e.g. answer 2 for Fallacy-Oversight).")

    def __init__(self, name: str, title: str, dataset: str, variants: VariantFn, expect_change: bool = False,
                 prompt: PromptFn = base_prompt):
        self.name, self.title, self.dataset, self.variants, self.prompt = name, title, dataset, variants, prompt
        self.expect_change = expect_change  # the perturbation ruins answer 1, so answer 2 becomes the correct pick
        self.truth = "2" if expect_change else "1"

    async def evaluate(self, judge, ctx) -> BiasResult:
        T, opts, repeats = ctx.T, ctx.options(self.name), ctx.cfg.cr_repeats

        async def run_item(row: dict) -> tuple[list, list, list]:
            orig_prompt = self.prompt(row, T)
            variants = self.variants(row, T, opts)
            outs = await asyncio.gather(*(ctx.ask(judge, orig_prompt, r) for r in range(repeats)),
                                        *(ctx.ask(judge, v.prompt) for v in variants))
            orig = [to_answer(parse_verdict(o), False) for o in outs[:repeats]]
            base = majority(orig)
            base_out = outs[orig.index(base)] if base in orig else outs[0]  # a repeat that gave the majority verdict
            pert = [to_answer(parse_verdict(o), v.swapped) for v, o in zip(variants, outs[repeats:])]
            samples = [self._sample(row["_idx"], v, out, base, p, base_out, len(orig))
                       for v, out, p in zip(variants, outs[repeats:], pert)]
            return Item(orig, [(base, p) for p in pert], consistency_pairs([orig])), samples

        done = await asyncio.gather(*(run_item(r) for r in ctx.data(self.dataset)))
        items = [item for item, _ in done]
        pairs = [pair for item in items for pair in item.pairs]
        value, ci = proportion([is_robust(o, p, self.expect_change) for o, p in pairs])
        cr = self._cr(items)
        drift_lo, drift_hi = bootstrap_ci(items, lambda xs: drift([p for x in xs for p in x.pairs]))
        return BiasResult(
            value=value, ci=ci, n=len(items),
            expected=expected_rr(cr, self.expect_change),
            effect=self._effect(items), effect_ci=bootstrap_ci(items, self._effect),
            stats={"cr": cr, "drift": drift(pairs), "drift_lo": drift_lo, "drift_hi": drift_hi},
            base_correctness=proportion([c == "1" for item in items for c in item.orig])[0],
            n_invalid=sum(c is None for item in items for c in [*item.orig, *(p for _, p in item.pairs)]),
            samples=[s for _, samples in done for s in samples],
        )

    @staticmethod
    def _cr(items: list[Item]) -> float:
        return proportion([is_robust(o, p) for item in items for o, p in item.cr_pairs])[0]

    def _effect(self, items: list[Item]) -> float:
        """RR - expected RR over these items (the bootstrap statistic)."""
        rr = proportion([is_robust(o, p, self.expect_change) for item in items for o, p in item.pairs])[0]
        return rr - expected_rr(self._cr(items), self.expect_change)

    def _sample(self, idx, v: Variant, out: str | None, orig: str | None, pert: str | None,
                orig_out: str | None, repeats: int = 1) -> dict:
        """Correct = the perturbed verdict picks the ground-truth answer (self.truth), whichever position it is shown in.
        Robustness (orig vs. pert agreement) is the bias metric, computed separately in evaluate(). base_status grades
        the unperturbed majority verdict, where answer 1 is always correct (also for Fallacy-Oversight)."""
        note = (f"variant: {v.name}" + (" (answers shown swapped: answer 1 is B)" if v.swapped else "")
                + f"; unperturbed verdict (majority of {repeats}): {'unparsable' if orig is None else f'answer {orig}'}")
        letter = "A" if (self.truth == "1") != v.swapped else "B"
        why = ("answer 1 was made logically chaotic, so answer 2 is now better" if self.expect_change
               else "the reference-preferred answer")
        desired = f"[[{letter}]] (answer {self.truth}, {why})"
        status = "invalid" if pert is None else "correct" if pert == self.truth else "wrong"
        base_status = "invalid" if orig is None else "correct" if orig == "1" else "wrong"
        return sample(idx, status, v.prompt, desired, out, note,
                      {"Unperturbed output (a repeat giving the majority verdict)": orig_out}, base_status)


# --- the 9 robustness panels, in Goal.png order -------------------------------------------------

def _position(row, T, opts):
    return [Variant("swapped", base_prompt(row, T, row["answer2"], row["answer1"]), swapped=True)]


def _verbosity(row, T, opts):
    return [Variant("longer", base_prompt(row, T, a2=row["answer2_longer"]))]


def _compassion_fade(row, T, opts):
    # Two distinct names drawn in random order, seeded per row so every judge sees the same pairing.
    names = opts.get("model_names", ["GPT-4o", "Claude-3.5-Sonnet"])
    if len(set(names)) < 2:
        raise ValueError("compassion_fade.model_names needs at least 2 distinct names")
    a, b = random.Random(f"compassion_fade|{row['_idx']}").sample(sorted(set(names)), 2)
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


for _name, _title, _ds, _fn, *_flags in [
    ("position", "Position", "alignment", _position),
    ("verbosity", "Verbosity", "verbosity", _verbosity),
    ("compassion_fade", "Compassion-Fade", "alignment", _compassion_fade),
    ("bandwagon", "Bandwagon", "alignment", _bandwagon),
    ("distraction", "Distraction", "alignment", _distraction),
    ("fallacy_oversight", "Fallacy-Oversight", "fallacy_oversight", _fallacy, True),
    ("authority", "Authority", "authority", _authority),
    ("sentiment", "Sentiment", "sentiment", _sentiment),
    ("diversity", "Diversity", "alignment", _diversity),
]:
    register(PerturbationBias(_name, _title, _ds, _fn, *_flags))
