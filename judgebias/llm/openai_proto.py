from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .base import ChatClient


class OpenAIProtocolClient(ChatClient):
    """Any OpenAI-compatible Chat Completions endpoint (OpenAI, DeepSeek, Qwen, vLLM, OpenRouter...)."""

    def __init__(self, api_key: str, base_url: str | None, token_param: str = "max_completion_tokens"):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=4, timeout=600)
        self._token_param = token_param

    async def complete(self, model, prompt, effort, max_tokens, params: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {self._token_param: max_tokens}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        kwargs.update(params)
        resp = await self._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return resp.choices[0].message.content or ""
