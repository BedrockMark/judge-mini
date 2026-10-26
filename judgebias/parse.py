"""Extract verdicts / ratings from judge output. The last match wins (explanations may quote the format)."""
from __future__ import annotations

import re

_VERDICT = re.compile(r"\[\[\s*([ABC])\s*\]\]")
_RATING = re.compile(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]")


def parse_verdict(text: str | None) -> str | None:
    if not text:
        return None
    found = _VERDICT.findall(text)
    return found[-1] if found else None


def parse_rating(text: str | None) -> float | None:
    if not text:
        return None
    found = _RATING.findall(text)
    if not found:
        return None
    value = float(found[-1])
    return value if 1 <= value <= 10 else None
