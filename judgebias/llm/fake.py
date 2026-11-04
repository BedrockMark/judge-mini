from __future__ import annotations

import hashlib
import random

from .base import ChatClient, Completion


# Fake thinking budget per effort, so offline cost plots show effort scaling.
EFFORT_TOKENS = {None: 50, "minimal": 50, "low": 200, "medium": 600, "high": 1500, "xhigh": 3000, "max": 6000}


class FakeClient(ChatClient):
    """Offline stand-in: deterministic pseudo-random verdicts/ratings keyed on (model, effort, prompt)."""

    async def complete(self, model, prompt, effort, max_tokens, params) -> Completion:
        seed = hashlib.sha256(f"{model}|{effort}|{prompt}".encode()).digest()
        rng = random.Random(seed)
        tokens = int(EFFORT_TOKENS.get(effort, 600) * rng.uniform(0.5, 1.5))
        if "[[Rating]]" in prompt:
            return Completion(f"Fake explanation. Rating: [[{rng.randint(5, 9)}]]", tokens)
        if '"[[A]]"' in prompt:
            # Mostly prefer A to mimic a judge with some position preference.
            return Completion(f"Fake explanation. [[{'A' if rng.random() < 0.7 else 'B'}]]", tokens)
        return Completion(f"Fake answer {rng.randint(0, 10**6)}.", tokens)
