from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from .base import ChatClient, Completion


class AnthropicProtocolClient(ChatClient):
    """Anthropic Messages API. `effort` -> output_config.effort; thinking stays adaptive (model default).

    No server-side refusal fallback on purpose: it would silently answer with a different model.
    """

    def __init__(self, api_key: str, base_url: str | None):
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=4, timeout=600)

    async def complete(self, model, prompt, effort, max_tokens, params: dict[str, Any]) -> Completion:
        kwargs: dict[str, Any] = {}
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}
        kwargs.update(params)
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        tokens = resp.usage.output_tokens if resp.usage else None  # includes thinking
        reasoning = "\n\n".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking)
        if resp.stop_reason == "refusal":
            return Completion("", tokens, reasoning=reasoning)
        return Completion("".join(b.text for b in resp.content if b.type == "text"), tokens, reasoning=reasoning)
