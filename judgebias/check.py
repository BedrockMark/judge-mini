"""Pre-flight check: config + keys, then one live judge call per judge (uncached) to surface errors early."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .biases import BIASES
from .config import Config, ConfigError, JudgeSpec, require_dataset, resolve_keys
from .llm import ChatClient, build_clients
from .parse import parse_verdict
from .prompts import load_templates


@dataclass
class CallCheck:
    judge: str
    ok: bool
    detail: str
    seconds: float


def _test_prompt(cfg: Config) -> str:
    T = load_templates(cfg.dataset_dir)
    return T.evaluate_ai_responses_no_tie("What is 2 + 2?", "2 + 2 = 4.", "2 + 2 = 5.")


async def _call_one(judge: JudgeSpec, client: ChatClient, prompt: str, max_tokens: int) -> CallCheck:
    start = time.perf_counter()
    try:
        text = await client.complete(judge.model, prompt, judge.effort, max_tokens, judge.params)
    except Exception as e:
        return CallCheck(judge.label, False, f"{type(e).__name__}: {e}", time.perf_counter() - start)
    elapsed = time.perf_counter() - start
    verdict = parse_verdict(text)
    if verdict is None:
        snippet = (text or "<empty response - refusal or token cap?>").strip().replace("\n", " ")[:200]
        return CallCheck(judge.label, False, f"no [[A]]/[[B]] verdict in response: {snippet}", elapsed)
    note = "" if verdict == "A" else " (picked the wrong answer; call works, but worth a look)"
    return CallCheck(judge.label, True, f"verdict [[{verdict}]]{note}", elapsed)


async def live_check(cfg: Config, keys: dict[str, str | None]) -> list[CallCheck]:
    clients = build_clients(cfg, keys)
    prompt = _test_prompt(cfg)
    return await asyncio.gather(*(_call_one(j, clients[j.provider], prompt, cfg.max_tokens) for j in cfg.judges))


def run_check(cfg: Config, live: bool = True) -> bool:
    """Print a report; return True only if everything passed. Config/key errors raise ConfigError."""
    keys = resolve_keys(cfg)
    require_dataset(cfg)
    biases = cfg.selected_biases()
    print(f"Config OK: {len(cfg.judges)} judge(s), {len(biases)} bias(es); keys found for {sorted(keys)}")
    for name in biases:
        for warning in BIASES[name].warnings(cfg.judges):
            print(f"  warning: {warning}")
    datasets = sorted({d for name in biases if (d := getattr(BIASES[name], "dataset", None))})
    if missing := [str(cfg.dataset_path(d)) for d in datasets if not cfg.dataset_path(d).is_file()]:
        raise ConfigError(f"Dataset files not found: {missing}")
    if not live:
        return True

    print("Test call per judge (uncached):")
    results = asyncio.run(live_check(cfg, keys))
    for r in results:
        status = "OK  " if r.ok else "FAIL"
        print(f"  [{status}] {r.judge:<28} {r.seconds:5.1f}s  {r.detail}")
    failed = [r for r in results if not r.ok]
    print("All judges reachable." if not failed else f"{len(failed)} judge(s) failed - fix before `run`.")
    return not failed
