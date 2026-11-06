from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .base import ChatClient, Completion


class OpenAIProtocolClient(ChatClient):
    """Any OpenAI-compatible Chat Completions endpoint (OpenAI, DeepSeek, Qwen, vLLM, OpenRouter...).

    `base_url` may list several comma-separated replicas of the same server (e.g. one vLLM per GPU);
    each request goes to the replica with the fewest requests in flight.
    """

    def __init__(self, api_key: str, base_url: str | None, token_param: str = "max_completion_tokens",
                 timeout: float = 600):
        urls = [u.strip() for u in base_url.split(",") if u.strip()] if base_url else [None]
        self._clients = [AsyncOpenAI(api_key=api_key, base_url=u, max_retries=4, timeout=timeout) for u in urls]
        self._inflight = [0] * len(self._clients)
        self._token_param = token_param

    @property
    def replicas(self) -> int:
        return len(self._clients)

    async def complete(self, model, prompt, effort, max_tokens, params: dict[str, Any]) -> Completion:
        kwargs: dict[str, Any] = {self._token_param: max_tokens}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        kwargs.update(params)
        i = min(range(len(self._clients)), key=self._inflight.__getitem__)
        self._inflight[i] += 1
        try:
            resp = await self._clients[i].chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], **kwargs
            )
        finally:
            self._inflight[i] -= 1
        tokens = resp.usage.completion_tokens if resp.usage else None  # includes reasoning tokens
        return Completion(resp.choices[0].message.content or "", tokens)
