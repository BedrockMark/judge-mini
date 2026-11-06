"""Config loading/validation and API-key resolution (.env + key-matching)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    pass


class ProviderSpec(BaseModel):
    protocol: Literal["openai", "anthropic", "fake"]
    base_url: str | None = None  # openai protocol: several comma-separated replicas are load-balanced
    base_url_env: str | None = None  # read base_url from this env variable instead (e.g. a vLLM job's address)
    api_key_env: str | None = None  # name of the variable in .env, never the key itself
    token_param: str = "max_completion_tokens"  # openai protocol only; some compat servers want max_tokens
    rpm: int | None = None  # client-side cap on requests per 60 s, shared by every judge on this provider (key)
    concurrency: int | None = None  # per-model in-flight cap per replica (base URL); default: global concurrency
    timeout: float = 600  # seconds per request (openai protocol), incl. time queued inside the server
    # A vllm_job.sh server: also record FLOPs per call (judgebias/flops.py), from $VLLM_MODEL_DIR/config.json and the
    # server's /metrics FLOP counter ($VLLM_TP GPUs per replica); vllm_job.sh sets both variables.
    flops: bool = False


class JudgeSpec(BaseModel):
    label: str
    provider: str
    model: str
    effort: str | None = None  # null = send no reasoning params
    color: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # extra request params, passed through verbatim
    group: str = ""  # config entry this judge came from; one entry with an effort list gives one group

    @model_validator(mode="after")
    def _default_group(self) -> "JudgeSpec":
        self.group = self.group or self.label
        return self


def expand_efforts(raw: dict) -> list[dict]:
    """`effort: [low, high]` -> one judge per effort, labelled "<label> (<effort>)", all in group <label>."""
    efforts = raw.get("effort")
    if not isinstance(efforts, list):
        return [raw]
    if not efforts:
        raise ConfigError(f"Judge '{raw.get('label')}' has an empty effort list")
    return [{**raw, "label": f"{raw.get('label')} ({e})", "effort": e, "group": raw.get("label")} for e in efforts]


class Config(BaseModel):
    dataset_dir: Path
    env_file: Path | None = Path(".env")
    results_dir: Path = Path("results")
    log_dir: Path | None = None  # default: <results_dir>/logs
    samples_per_bias: int = 50
    seed: int = 0
    concurrency: int = 8  # max in-flight requests per (provider, model)
    cr_repeats: int = 2
    max_tokens: int = 16000
    providers: dict[str, ProviderSpec]
    judges: list[JudgeSpec]
    biases: Literal["all"] | list[str] = "all"
    bias_options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("judges", mode="before")
    @classmethod
    def _expand_efforts(cls, judges: Any) -> Any:
        if not isinstance(judges, list):
            return judges
        return [x for j in judges for x in (expand_efforts(j) if isinstance(j, dict) else [j])]

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
    for field in ("dataset_dir", "env_file", "results_dir", "log_dir"):
        value = getattr(cfg, field)
        if value is not None and not value.is_absolute():
            setattr(cfg, field, base / value)
    if cfg.log_dir is None:
        cfg.log_dir = cfg.results_dir / "logs"
    return cfg


SNAPSHOT_HEADER = "# judgebias config snapshot of "
PATH_FIELDS = ("dataset_dir", "env_file", "results_dir", "log_dir")


def snapshot_config(src: str | Path, dest: str | Path) -> Path:
    """Copy a config as it is now, for a job that reads it later: editing the original then can't change a queued
    job, so several differently configured jobs can be submitted one after another. Relative paths are made
    absolute (they are relative to the config's own folder), so the copy works wherever it is stored."""
    src, dest = Path(src), Path(dest)
    cfg = load_config(src)  # also validates, so a broken config fails at submission
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    for field in PATH_FIELDS:
        value = getattr(cfg, field)
        raw[field] = None if value is None else str(value)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"{SNAPSHOT_HEADER}{config_origin(src)} ({datetime.now():%Y-%m-%d %H:%M:%S})\n"
                    "# Paths made absolute. Jobs submitted with this copy read it, not the original.\n"
                    + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dest


def config_origin(path: str | Path) -> str:
    """The original config a snapshot was taken from (the path itself if it is not a snapshot)."""
    path = Path(path)
    first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    return first[len(SNAPSHOT_HEADER):].rsplit(" (", 1)[0] if first.startswith(SNAPSHOT_HEADER) else str(path.resolve())


def require_dataset(cfg: Config) -> None:
    """The dataset is a git submodule; a plain `git clone` leaves its folder empty."""
    if not (cfg.dataset_dir / "promptTemplate.py").exists():
        raise ConfigError(f"Dataset not found in {cfg.dataset_dir}; fetch the submodule with "
                          "`git submodule update --init --remote`")


def resolve_keys(cfg: Config) -> dict[str, str | None]:
    """Load .env and return {provider_name: api_key} for every provider a judge uses.

    Also fills base_url of providers that set base_url_env.
    Raises ConfigError listing every missing key/URL before any API call is made.
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
        if spec.base_url_env:
            url = os.environ.get(spec.base_url_env, "").strip()
            if url:
                spec.base_url = url
            else:
                missing.append(f"{spec.base_url_env} (base URL of provider '{name}')")
        if not spec.api_key_env:
            missing.append(f"provider '{name}' has no api_key_env")
            continue
        key = os.environ.get(spec.api_key_env, "").strip()
        if not key:
            missing.append(f"{spec.api_key_env} (provider '{name}')")
            continue
        keys[name] = key
    if missing:
        raise ConfigError(f"Missing API keys/URLs in {cfg.env_file}: " + ", ".join(missing))
    return keys
