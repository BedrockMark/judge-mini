from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ChatClient(ABC):
    """One protocol adapter. Implementations translate `effort` into their own request params."""

    @abstractmethod
    async def complete(
        self, model: str, prompt: str, effort: str | None, max_tokens: int, params: dict[str, Any]
    ) -> str:
        """Return the final text answer ('' if the model produced none, e.g. a refusal)."""
