from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Completion(str):
    """The text answer, plus the output tokens it cost (thinking included); None when the API did not say."""

    output_tokens: int | None

    def __new__(cls, text: str, output_tokens: int | None = None) -> "Completion":
        obj = super().__new__(cls, text)
        obj.output_tokens = output_tokens
        return obj


class ChatClient(ABC):
    """One protocol adapter. Implementations translate `effort` into their own request params."""

    replicas: int = 1  # identical servers behind this client; the per-model concurrency cap scales with it

    @abstractmethod
    async def complete(
        self, model: str, prompt: str, effort: str | None, max_tokens: int, params: dict[str, Any]
    ) -> str:
        """Return the final text answer ('' if the model produced none, e.g. a refusal), ideally as a
        Completion carrying the output token count; a plain str counts as unknown usage."""
