"""Writes results/examples/<bias>/<judge>/{correct,wrong,invalid}.md: up to `k` examples each, with counts on top.
Biases whose samples carry a `base_status` (the unbiased prompt's verdict: robustness panels and CoT) also get
bias_better_case.md (unbiased prompt wrong, biased prompt correct) and bias_worse_case.md (the reverse), explained by
the bias's `case_note`. Examples are
drawn from distinct dataset rows (so distinct unbiased prompts) first; a row repeats only when there are too few."""
from __future__ import annotations

import re
from pathlib import Path

STATUSES = ("correct", "wrong", "invalid")
# case -> (base_status, status): what the unbiased and the biased prompt got
BIAS_CASES = {"bias_better_case": ("wrong", "correct"), "bias_worse_case": ("correct", "wrong")}


def _in_case(x: dict, case: str) -> bool:
    if case in BIAS_CASES:
        return (x.get("base_status"), x["status"]) == BIAS_CASES[case]
    return x["status"] == case


def _pick(xs: list[dict], k: int) -> list[dict]:
    """Up to k samples, one per dataset row first, then repeats in order to fill up."""
    seen, first, rest = set(), [], []
    for x in xs:
        (rest if x["idx"] in seen else first).append(x)
        seen.add(x["idx"])
    return (first + rest)[:k]


def _fence(text: str | None) -> str:
    """Fenced block that survives backticks inside model output."""
    if text is None:
        return "*(no response: the API call failed)*"
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def _has_base(samples: list[dict]) -> bool:
    return any(x.get("base_status") for x in samples)


def render(judge, bias, status: str, samples: list[dict], k: int) -> str:
    cases = [*STATUSES, *(BIAS_CASES if _has_base(samples) else ())]
    counts = {c: sum(_in_case(x, c) for x in samples) for c in cases}
    matching = [x for x in samples if _in_case(x, status)]
    shown = _pick(matching, k)
    rows = len({x["idx"] for x in samples})
    extra_cols = cases[len(STATUSES):]
    head = ["Judge", "Model", "Effort", "Bias", "Correct", "Wrong", "Invalid",
            *(c.replace("_", " ").capitalize() for c in extra_cols), "Total", "Rows"]
    cells = [judge.label, judge.model, judge.effort, bias.name, *(counts[c] for c in cases), len(samples), rows]
    lines = [
        f"# {bias.title}: {status} examples for {judge.label}",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        "| " + " | ".join(map(str, cells)) + " |",
        "",
        "Total counts judged samples; a bias with several perturbations per dataset row "
        "(e.g. sentiment tones, authority kinds, diversity groups) yields several samples per row.",
    ]
    if extra_cols:
        lines += ["", bias.case_note]
    lines += ["", f"Showing {len(shown)} of {counts[status]} {status} examples "
              f"(from {len({x['idx'] for x in shown})} distinct dataset rows; {len({x['idx'] for x in matching})} available)."]
    for i, x in enumerate(shown, 1):
        lines += ["", "---", "", f"## Example {i} (dataset row {x['idx']})"]
        if x["note"]:
            lines += ["", f"*{x['note']}*"]
        if x.get("base_status"):
            lines += ["", f"*unbiased prompt: {x['base_status']}; biased prompt: {x['status']}*"]
        lines += ["", "### Input", "", _fence(x["input"]),
                  "", "### Desired output", "", x["desired"]]
        if x.get("thinking"):
            lines += ["", "### Model thinking", "", _fence(x["thinking"])]
        lines += ["", "### Model output", "", _fence(x["output"])]
        for title, text in x["extra"].items():
            lines += ["", f"### {title}", "", _fence(text)]
    return "\n".join(lines) + "\n"


def write_examples(out_dir: Path, judge, bias, samples: list[dict], k: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for status in [*STATUSES, *(BIAS_CASES if _has_base(samples) else ())]:
        (out_dir / f"{status}.md").write_text(render(judge, bias, status, samples, k), encoding="utf-8")
