"""Config loading/validation and API-key resolution (.env + key-matching)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigError(ValueError):
    pass


class ProviderSpec(BaseModel):
    protocol: Literal["openai", "anthropic", "fake"]
    base_url: str | None = None
    api_key_env: str | None = None  # name of the variable in .env, never the key itself
    token_param: str = "max_completion_tokens"  # openai protocol only; some compat servers want max_tokens


class JudgeSpec(BaseModel):
    label: str
    provider: str
    model: str
    effort: str | None = None  # null = send no reasoning params
    color: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # extra request params, passed through verbatim


class Config(BaseModel):
    dataset_dir: Path
    env_file: Path | None = Path(".env")
    results_dir: Path = Path("results")
    samples_per_bias: int = 50
    seed: int = 0
    concurrency: int = 8
    cr_repeats: int = 2
    max_tokens: int = 16000
    providers: dict[str, ProviderSpec]
    judges: list[JudgeSpec]
    biases: Literal["all"] | list[str] = "all"
    bias_options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_refs(self) -> "Config":
        labels = [j.label for j in self.judges]
        if len(set(labels)) != len(labels):
            raise ConfigError(f"Judge labels must be unique: {labels}")
        for j in self.judges:
            if j.provider not in self.providers:
                raise ConfigError(f"Judge '{j.label}' uses unknown provider '{j.provider}'")
        if self.cr_repeats < 2:
            raise ConfigError("cr_repeats must be >= 2 to measure consistency")
        return self

    def selected_biases(self) -> list[str]:
        from .biases import BIASES
        names = list(BIASES) if self.biases == "all" else list(self.biases)
        unknown = [b for b in names if b not in BIASES]
        if unknown:
            raise ConfigError(f"Unknown biases {unknown}; available: {list(BIASES)}")
        return names


def parse_config(raw: dict) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(str(e)) from None


def load_config(path: str | Path) -> Config:
    path = Path(path)
    cfg = parse_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    base = path.resolve().parent
    for field in ("dataset_dir", "env_file", "results_dir"):
        value = getattr(cfg, field)
        if value is not None and not value.is_absolute():
            setattr(cfg, field, base / value)
    return cfg


def resolve_keys(cfg: Config) -> dict[str, str | None]:
    """Load .env and return {provider_name: api_key} for every provider a judge uses.

    Raises ConfigError listing every missing key before any API call is made.
    """
    if cfg.env_file is not None and cfg.env_file.exists():
        load_dotenv(cfg.env_file, override=False)
    keys: dict[str, str | None] = {}
    missing = []
    for name in sorted({j.provider for j in cfg.judges}):
        spec = cfg.providers[name]
        if spec.protocol == "fake":
            keys[name] = None
            continue
        if not spec.api_key_env:
            missing.append(f"provider '{name}' has no api_key_env")
            continue
        key = os.environ.get(spec.api_key_env, "").strip()
        if not key:
            missing.append(f"{spec.api_key_env} (provider '{name}')")
            continue
        keys[name] = key
    if missing:
        raise ConfigError(f"Missing API keys in {cfg.env_file}: " + ", ".join(missing))
    return keys
