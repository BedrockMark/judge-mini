"""FLOPs per call for judges served by our own vLLM job (vllm_job.sh starts it with --enable-mfu-metrics).

vLLM only reports a server-wide counter (`vllm:estimated_flops_per_gpu_total` on /metrics), summed over every
request; a run sends all judge x bias pairs to the same server at once, so the counter cannot be split by itself.
Instead each call is estimated with the same per-step formulas vLLM uses (vllm/v1/metrics/perf.py) from the model's
config.json and the call's prompt / output token counts, and the runner scales those estimates so that they add up
to the counter's increase over the run. The shares come from the formulas, the total from vLLM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

METRIC = "vllm:estimated_flops_per_gpu_total"


def _sum_min(a: int, b: int, window: int | None) -> int:
    """sum(min(k, window) for k in a..b), 0 if b < a."""
    if b < a:
        return 0
    if window is None or b <= window:
        return (a + b) * (b - a + 1) // 2
    if a > window:
        return window * (b - a + 1)
    return (a + window) * (window - a + 1) // 2 + window * (b - window)


@dataclass
class FlopModel:
    """Shapes from a Hugging Face config.json; counts are for the whole model (all GPUs), multiply-add = 2."""

    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    windows: list[int | None]  # per layer: sliding window, or None for full attention
    ffn_per_token: float  # FLOPs of the dense / routed + shared expert MLPs, all layers

    @classmethod
    def from_config(cls, cfg: dict) -> "FlopModel":
        cfg = cfg if "hidden_size" in cfg else cfg.get("text_config", cfg)  # multimodal configs nest the LM
        D, L = cfg["hidden_size"], cfg["num_hidden_layers"]
        q = cfg["num_attention_heads"]
        kv = cfg.get("num_key_value_heads") or q
        d = cfg.get("head_dim") or D // q
        types = cfg.get("layer_types") or ["full_attention"] * L
        windows = [cfg.get("sliding_window") if t == "sliding_attention" else None for t in types]
        experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
        if experts:  # every layer MoE (gpt-oss, Mixtral, Qwen3-MoE): top-k routed experts + any shared ones
            inter = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
            topk = cfg.get("num_experts_per_tok") or cfg.get("experts_per_token")
            shared = (cfg.get("n_shared_experts") or 0) * inter + (cfg.get("shared_expert_intermediate_size") or 0)
            ffn = 2 * D * 3 * (topk * inter + shared) * L
        else:
            ffn = 2 * D * 3 * cfg["intermediate_size"] * L
        return cls(D, q, kv, d, cfg["vocab_size"], windows, ffn)

    @classmethod
    def load(cls, model_dir: str | Path) -> "FlopModel":
        return cls.from_config(json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")))

    def call_flops(self, prompt_tokens: int, output_tokens: int) -> float:
        """One request as vLLM steps through it: the prompt in one prefill step (context = prompt), then one decode
        step per further output token (context = position). The last output token is never fed back."""
        p, o = prompt_tokens, max(output_tokens, 1)
        tokens = p + o - 1
        D, q, kv, d = self.hidden, self.heads, self.kv_heads, self.head_dim
        proj = (2 * D * (q + 2 * kv) * d + 2 * D * q * d) * len(self.windows)  # QKV + output projections
        # perf.py bills a prefill chunk as num_tokens x context_len (not the causal triangle), clamped to the window
        token_context = sum(p * min(p, w or p) + _sum_min(p + 1, p + o - 1, w) for w in self.windows)
        attn = 4 * q * d * token_context  # QK^T and AV
        logits = 2 * D * self.vocab * o  # last prefill token + every decode token
        return tokens * (proj + self.ffn_per_token) + attn + logits


def parse_counter(text: str, metric: str = METRIC) -> float | None:
    """Sum of a counter over all its label sets (one per engine) in Prometheus text format; None if absent."""
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith(metric) and line[len(metric):len(metric) + 1] in ("{", " ")]
    return sum(values) if values else None


async def measured_flops(base_urls: list[str], api_key: str | None, gpus_per_replica: int) -> float | None:
    """vLLM's FLOP counter summed over all replicas (each GPU of a replica does the per-GPU amount); None when a
    replica has no counter (server started without --enable-mfu-metrics) or cannot be reached."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def get(url: str) -> str:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as resp:
            return resp.read().decode("utf-8")

    total = 0.0
    for url in base_urls:
        root = url.rstrip("/").removesuffix("/v1")
        try:
            text = await asyncio.to_thread(get, f"{root}/metrics")
        except OSError as e:  # URLError / HTTPError / timeouts
            log.warning("Could not read %s/metrics: %s", root, e)
            return None
        value = parse_counter(text)
        if value is None:
            log.warning("%s/metrics has no %s (vLLM started without --enable-mfu-metrics?)", root, METRIC)
            return None
        total += value * gpus_per_replica
    return total
