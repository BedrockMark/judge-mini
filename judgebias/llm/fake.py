from __future__ import annotations

import hashlib
import random

from .base import ChatClient


class FakeClient(ChatClient):
    """Offline stand-in: deterministic pseudo-random verdicts/ratings keyed on (model, effort, prompt)."""

    async def complete(self, model, prompt, effort, max_tokens, params) -> str:
        seed = hashlib.sha256(f"{model}|{effort}|{prompt}".encode()).digest()
        rng = random.Random(seed)
        if "[[Rating]]" in prompt:
            return f"Fake explanation. Rating: [[{rng.randint(5, 9)}]]"
        if '"[[A]]"' in prompt:
            # Mostly prefer A to mimic a judge with some position preference.
            return f"Fake explanation. [[{'A' if rng.random() < 0.7 else 'B'}]]"
        return f"Fake answer {rng.randint(0, 10**6)}."
