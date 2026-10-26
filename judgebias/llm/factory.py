from __future__ import annotations

from ..config import Config
from .anthropic_proto import AnthropicProtocolClient
from .base import ChatClient
from .fake import FakeClient
from .openai_proto import OpenAIProtocolClient


def build_clients(cfg: Config, keys: dict[str, str | None]) -> dict[str, ChatClient]:
    """One client per provider actually used by a judge; `keys` comes from config.resolve_keys."""
    clients: dict[str, ChatClient] = {}
    for name in keys:
        spec = cfg.providers[name]
        if spec.protocol == "openai":
            clients[name] = OpenAIProtocolClient(keys[name], spec.base_url, spec.token_param)
        elif spec.protocol == "anthropic":
            clients[name] = AnthropicProtocolClient(keys[name], spec.base_url)
        else:
            clients[name] = FakeClient()
    return clients
