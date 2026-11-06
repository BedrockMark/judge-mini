"""Writes results/examples/<bias>/<judge>/{correct,wrong,invalid}.md: up to `k` examples each, with counts on top."""
from __future__ import annotations

import re
from pathlib import Path

STATUSES = ("correct", "wrong", "invalid")


def _fence(text: str | None) -> str:
    """Fenced block that survives backticks inside model output."""
    if text is None:
        return "*(no response: the API call failed)*"
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def render(judge, bias, status: str, samples: list[dict], k: int) -> str:
    counts = {s: sum(x["status"] == s for x in samples) for s in STATUSES}
    shown = [x for x in samples if x["status"] == status][:k]
    rows = len({x["idx"] for x in samples})
    lines = [
        f"# {bias.title}: {status} examples for {judge.label}",
        "",
        "| Judge | Model | Effort | Bias | Correct | Wrong | Invalid | Total | Rows |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {judge.label} | {judge.model} | {judge.effort} | {bias.name} | {counts['correct']} | {counts['wrong']} "
        f"| {counts['invalid']} | {len(samples)} | {rows} |",
        "",
        "Total counts judged samples; a bias with several perturbations per dataset row "
        "(e.g. sentiment tones, authority kinds, diversity groups) yields several samples per row.",
        "",
        f"Showing {len(shown)} of {counts[status]} {status} examples.",
    ]
    for i, x in enumerate(shown, 1):
        lines += ["", "---", "", f"## Example {i} (dataset row {x['idx']})"]
        if x["note"]:
            lines += ["", f"*{x['note']}*"]
        lines += ["", "### Input", "", _fence(x["input"]),
                  "", "### Desired output", "", x["desired"],
                  "", "### Model output", "", _fence(x["output"])]
        for title, text in x["extra"].items():
            lines += ["", f"### {title}", "", _fence(text)]
    return "\n".join(lines) + "\n"


def write_examples(out_dir: Path, judge, bias, samples: list[dict], k: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for status in STATUSES:
        (out_dir / f"{status}.md").write_text(render(judge, bias, status, samples, k), encoding="utf-8")
