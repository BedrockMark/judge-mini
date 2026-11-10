import asyncio
import colorsys
import json
import math
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from judgebias.biases import BIASES
from judgebias.biases.base import sample
from judgebias.biases.pairwise import Variant, base_prompt, to_answer
from judgebias.config import Config, ConfigError, load_config, parse_config, resolve_keys
from judgebias.context import Context
from judgebias.llm import ResponseCache, build_clients
from judgebias.llm.anthropic_proto import AnthropicProtocolClient
from judgebias.llm.fake import FakeClient
from judgebias.llm.openai_proto import OpenAIProtocolClient
from judgebias.metrics import (accuracy, bootstrap_ci, consistency_pairs, consistency_rate, drift, error_rate, expected_rr,
                               majority, mean_ci, proportion, robustness_rate, wilson)
from judgebias.runner import SUMMARY_COLUMNS
from judgebias.parse import parse_rating, parse_verdict

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "LLM-Judge-Bias-Dataset"


def make_cfg(tmp_path, **overrides) -> Config:
    raw = {
        "dataset_dir": DATASET,
        "env_file": tmp_path / ".env",
        "results_dir": tmp_path / "results",
        "samples_per_bias": 3,
        "providers": {
            "anth": {"protocol": "anthropic", "api_key_env": "TEST_ANTH_KEY"},
            "oai": {"protocol": "openai", "base_url": "https://example.com/v1", "api_key_env": "TEST_OAI_KEY"},
            "fake": {"protocol": "fake"},
        },
        "judges": [
            {"label": "a", "provider": "fake", "model": "m1", "effort": "low"},
            {"label": "b", "provider": "fake", "model": "m2"},
        ],
    }
    raw.update(overrides)
    return parse_config(raw)


# --- parsing & metrics --------------------------------------------------------------------------

def test_parse_verdict_takes_last_match():
    assert parse_verdict('Format is "[[A]]" or "[[B]]". Final: [[B]]') == "B"
    assert parse_verdict("no verdict") is None
    assert parse_verdict(None) is None


def test_parse_rating():
    assert parse_rating("Rating: [[7]]") == 7
    assert parse_rating("Rating: [[ 8.5 ]]") == 8.5
    assert parse_rating("Rating: [[42]]") is None
    assert parse_rating("") is None


def test_to_answer_maps_positions_back():
    assert to_answer("A", False) == "1" and to_answer("B", False) == "2"
    assert to_answer("A", True) == "2" and to_answer("B", True) == "1"
    assert to_answer("C", False) is None and to_answer(None, True) is None


def test_metrics():
    assert robustness_rate([("1", "1"), ("1", "2"), (None, None), ("2", "2")]) == 0.5
    assert consistency_rate([["1", "1"], ["1", "2"], [None, None]]) == pytest.approx(1 / 3)  # 2 repeats: the paper's CR
    assert accuracy(["1", "2", None, "1"], "1") == 0.5
    assert error_rate([(8, 8), (6, 8), (None, 8)]) == pytest.approx(0.125)
    assert math.isnan(robustness_rate([]))


def test_intervals_and_drift():
    lo, hi = wilson(90, 100)
    assert (lo, hi) == (pytest.approx(0.825, abs=0.001), pytest.approx(0.945, abs=0.001))  # textbook Wilson, p=0.9 n=100
    assert wilson(100, 100) == (pytest.approx(0.963, abs=0.001), pytest.approx(1.0))  # not degenerate at p=1
    assert all(math.isnan(v) for v in wilson(0, 0))
    assert proportion([True, False]) == (0.5, wilson(1, 2))
    m, (a, b) = mean_ci([0.1, 0.3, 0.2])
    assert m == pytest.approx(0.2) and a < 0.2 < b and mean_ci([0.5]) == (0.5, (0.5, 0.5))
    # drift: net share of verdicts moving 1 -> 2; symmetric flips cancel, invalid sides do not move
    assert drift([("1", "2"), ("1", "2"), ("2", "1"), ("1", "1"), (None, "2")]) == pytest.approx(0.2)
    assert drift([("1", "2"), ("2", "1")]) == 0 and math.isnan(drift([]))


def test_majority_baseline_and_leave_one_out_consistency():
    assert majority(["1", "2", "2"]) == "2" and majority(["2", None, "1", "1"]) == "1"
    assert majority(["1", "2", None]) == "1" and majority([None, "2", "1"]) == "2"  # a tie: the first valid repeat
    assert majority([None, None]) is None
    # each repeat vs the majority of the others: 3 of 4 agree -> 3 pairs agree, the odd one disagrees
    assert consistency_pairs([["1", "1", "1", "2"]]) == [("1", "1"), ("1", "1"), ("1", "1"), ("1", "2")]
    assert consistency_rate([["1", "1", "1", "2"]]) == 0.75
    assert consistency_rate([["1", "1", "1", "1"], ["1", "1", "2", "2"]]) == pytest.approx(0.5)  # 2-2 flips every pair
    assert math.isnan(consistency_rate([["1"]]))  # a single repeat has no pair
    # a perturbation with no effect scores CR; for Fallacy-Oversight (a change is robust) 1 - CR
    assert expected_rr(0.8) == 0.8 and expected_rr(0.8, expect_change=True) == pytest.approx(0.2)


def test_bootstrap_ci_brackets_the_statistic_and_is_reproducible():
    items = [1] * 80 + [0] * 20
    lo, hi = bootstrap_ci(items, lambda xs: sum(xs) / len(xs))
    assert lo < 0.8 < hi and 0.7 < lo and hi < 0.9
    assert bootstrap_ci(items, lambda xs: sum(xs) / len(xs)) == (lo, hi)  # seeded
    assert all(math.isnan(v) for v in bootstrap_ci([], lambda xs: 0.0))
    assert all(math.isnan(v) for v in bootstrap_ci([1], lambda xs: math.nan))  # NaN resamples are dropped


# --- config & key matching ----------------------------------------------------------------------

def test_unknown_provider_rejected(tmp_path):
    with pytest.raises(ConfigError):
        make_cfg(tmp_path, judges=[{"label": "x", "provider": "nope", "model": "m"}])


def test_duplicate_labels_rejected(tmp_path):
    j = {"label": "x", "provider": "fake", "model": "m"}
    with pytest.raises(ConfigError):
        make_cfg(tmp_path, judges=[j, j])


def test_missing_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ANTH_KEY", raising=False)
    cfg = make_cfg(tmp_path, judges=[{"label": "x", "provider": "anth", "model": "m"}])
    with pytest.raises(ConfigError, match="TEST_ANTH_KEY"):
        resolve_keys(cfg)


def test_keys_loaded_from_env_file_and_matched_to_protocol(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ANTH_KEY", raising=False)
    monkeypatch.delenv("TEST_OAI_KEY", raising=False)
    (tmp_path / ".env").write_text("TEST_ANTH_KEY=sk-anth\nTEST_OAI_KEY=sk-oai\n")
    cfg = make_cfg(tmp_path, judges=[
        {"label": "x", "provider": "anth", "model": "m"},
        {"label": "y", "provider": "oai", "model": "m", "effort": "high"},
    ])
    keys = resolve_keys(cfg)
    assert keys == {"anth": "sk-anth", "oai": "sk-oai"}
    clients = build_clients(cfg, keys)
    assert isinstance(clients["anth"], AnthropicProtocolClient)
    assert isinstance(clients["oai"], OpenAIProtocolClient)


def test_unused_provider_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ANTH_KEY", raising=False)
    cfg = make_cfg(tmp_path)  # only fake judges; 'anth' is defined but unused
    assert resolve_keys(cfg) == {"fake": None}


def test_base_url_read_from_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLLM_KEY", "k")
    monkeypatch.delenv("TEST_VLLM_URL", raising=False)
    providers = {"vllm": {"protocol": "openai", "base_url_env": "TEST_VLLM_URL", "api_key_env": "TEST_VLLM_KEY"}}
    cfg = make_cfg(tmp_path, providers=providers, judges=[{"label": "x", "provider": "vllm", "model": "m"}])
    with pytest.raises(ConfigError, match="TEST_VLLM_URL"):
        resolve_keys(cfg)
    monkeypatch.setenv("TEST_VLLM_URL", "http://node:1234/v1")
    resolve_keys(cfg)
    assert cfg.providers["vllm"].base_url == "http://node:1234/v1"


def test_model_filter_keeps_only_served_model(tmp_path, capsys):
    import yaml
    from judgebias.__main__ import main
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": str(tmp_path / "results"),
           "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1", "effort": ["low", "high"]},
                      {"label": "b", "provider": "fake", "model": "m2"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    base = ["--config", str(path), "--log-dir", str(tmp_path / "logs")]
    assert main(["check", *base, "--model", "m1"]) == 0
    assert "2 judge(s)" in capsys.readouterr().out
    assert main(["check", *base, "--model", "nope"]) == 2


def test_priority_gate_releases_lowest_priority_first():
    from judgebias.llm import PriorityGate

    async def main():
        gate, order = PriorityGate(1), []
        await gate.acquire(0)  # occupy the only slot so the rest queue up

        async def waiter(name, prio):
            await gate.acquire(prio)
            order.append(name)
            gate.release()

        tasks = [asyncio.create_task(waiter(n, p)) for n, p in [("low1", 2), ("high", 0), ("low2", 2), ("mid", 1)]]
        await asyncio.sleep(0)
        tasks[2].cancel()  # a cancelled waiter must not swallow a slot
        await asyncio.sleep(0)
        gate.release()
        await asyncio.gather(*tasks, return_exceptions=True)
        return order

    assert asyncio.run(main()) == ["high", "mid", "low1"]


def test_openai_client_spreads_requests_over_replicas():
    client = OpenAIProtocolClient("k", "http://a/v1, http://b/v1", timeout=7200)
    assert client.replicas == 2
    assert [str(c.base_url) for c in client._clients] == ["http://a/v1/", "http://b/v1/"]
    assert client._clients[0].timeout == 7200
    seen = []

    class Replica:
        def __init__(self, name):
            self.chat = self.completions = self
            self.name = name

        async def create(self, **kw):
            seen.append(self.name)
            await asyncio.sleep(0.01)
            msg = type("M", (), {"content": "[[A]]"})
            return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": None})

    client._clients = [Replica("a"), Replica("b")]

    async def main():
        await asyncio.gather(*(client.complete("m", "p", None, 10, {}) for _ in range(4)))

    asyncio.run(main())
    assert sorted(seen) == ["a", "a", "b", "b"]  # least-in-flight: never 3 on one replica


def test_concurrency_scales_with_replicas_and_high_effort_goes_first(tmp_path):
    class Probe(FakeClient):
        replicas = 2

        def __init__(self):
            self.inflight = self.peak = 0
            self.efforts = []

        async def complete(self, model, prompt, effort, max_tokens, params):
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            self.efforts.append(effort)
            await asyncio.sleep(0.01)
            self.inflight -= 1
            return "[[A]]"

    cfg = make_cfg(tmp_path, concurrency=99,
                   providers={"fake": {"protocol": "fake", "concurrency": 1}},
                   judges=[{"label": "j", "provider": "fake", "model": "m", "effort": ["low", "high"]}])
    client = Probe()
    cache = ResponseCache(tmp_path / "c.jsonl")
    ctx = Context(cfg, {"fake": client}, cache)
    low, high = cfg.judges

    async def main():
        await asyncio.gather(*(ctx.ask(j, f"p{i}") for i in range(4) for j in (low, high)))

    asyncio.run(main())
    cache.close()
    assert client.peak == 2  # provider concurrency (1) x replicas (2), not the global 99
    assert client.efforts[:2] == ["low", "high"]  # first two take the free slots in arrival order
    assert client.efforts[2:5] == ["high"] * 3  # then every queued high-effort call before the low ones


def test_repo_configs_load():
    for name in ("config.yaml", "config.fake.yaml", "config.vllm.yaml"):
        cfg = load_config(ROOT / name)
        assert cfg.dataset_dir.exists()
        assert cfg.selected_biases()


# --- biases on real dataset rows ----------------------------------------------------------------

class RecordingClient(FakeClient):
    def __init__(self):
        self.prompts = []

    async def complete(self, model, prompt, effort, max_tokens, params):
        self.prompts.append(prompt)
        return await super().complete(model, prompt, effort, max_tokens, params)


def test_twelve_biases_registered_in_figure_order():
    assert list(BIASES) == [
        "position", "verbosity", "compassion_fade", "bandwagon", "distraction", "fallacy_oversight",
        "authority", "sentiment", "diversity", "cot", "self_enhancement", "refinement_aware",
    ]


@pytest.mark.parametrize("bias_name", list(BIASES))
def test_every_bias_runs_on_real_rows(tmp_path, bias_name):
    cfg = make_cfg(tmp_path)
    client = RecordingClient()
    cache = ResponseCache(tmp_path / "cache.jsonl")
    ctx = Context(cfg, {"fake": client}, cache)
    res = asyncio.run(BIASES[bias_name].evaluate(cfg.judges[0], ctx))
    cache.close()
    assert res.n == 3
    assert 0 <= res.value <= 1.5
    assert client.prompts and all(p.strip() for p in client.prompts)
    assert not any("{" + "question}" in p for p in client.prompts)  # templates fully formatted


def test_position_swaps_answers():
    from judgebias.prompts import load_templates
    T = load_templates(DATASET)
    row = {"question": "Q", "answer1": "FIRST", "answer2": "SECOND"}
    p = base_prompt(row, T, row["answer2"], row["answer1"])
    assert p.index("SECOND") < p.index("FIRST")


def test_compassion_fade_samples_two_names_in_random_order():
    from judgebias.biases.pairwise import _compassion_fade
    from judgebias.prompts import load_templates
    T = load_templates(DATASET)
    names = ["M1", "M2", "M3", "M4"]
    pairs = [_compassion_fade({"_idx": i, "question": "Q", "answer1": "x", "answer2": "y"}, T,
                              {"model_names": names})[0].name.split("|") for i in range(40)]
    assert all(a != b and a in names and b in names for a, b in pairs)
    assert len({(a, b) for a, b in pairs}) > 6  # several pairs, both orders
    again = _compassion_fade({"_idx": 0, "question": "Q", "answer1": "x", "answer2": "y"}, T, {"model_names": names})
    assert again[0].name.split("|") == pairs[0]  # deterministic per row
    with pytest.raises(ValueError):
        _compassion_fade({"_idx": 0, "question": "Q", "answer1": "x", "answer2": "y"}, T, {"model_names": ["M1"]})


def test_cache_prevents_repeat_calls(tmp_path):
    cfg = make_cfg(tmp_path)
    for expected_calls in (None, 0):
        client = RecordingClient()
        cache = ResponseCache(tmp_path / "cache.jsonl")
        ctx = Context(cfg, {"fake": client}, cache)
        asyncio.run(BIASES["position"].evaluate(cfg.judges[0], ctx))
        cache.close()
        if expected_calls is None:
            assert client.prompts
        else:
            assert len(client.prompts) == expected_calls


@pytest.mark.parametrize("bias_name", [b for b, x in BIASES.items() if hasattr(x, "variants")])
def test_variants_build_for_every_dataset_row(bias_name):
    """Full-dataset sweep: no row may crash a variant builder (some rows lack perturbed fields)."""
    from judgebias.data import load_dataset
    from judgebias.prompts import load_templates
    bias, T = BIASES[bias_name], load_templates(DATASET)
    rows = load_dataset(DATASET, bias.dataset)
    built = [bias.variants(r, T, {}) for r in rows]
    assert sum(len(v) > 0 for v in built) >= 0.9 * len(rows)


# --- live check ---------------------------------------------------------------------------------

class _Raising(FakeClient):
    async def complete(self, *a, **k):
        raise RuntimeError("401 invalid x-api-key")


class _Rambling(FakeClient):
    async def complete(self, *a, **k):
        return "I think both are fine."


@pytest.mark.parametrize("client,ok,needle", [
    (FakeClient(), True, "verdict"),
    (_Raising(), False, "RuntimeError: 401 invalid x-api-key"),
    (_Rambling(), False, "no [[A]]/[[B]] verdict"),
])
def test_live_check_reports_each_outcome(tmp_path, client, ok, needle):
    from judgebias.check import _call_one, _test_prompt
    cfg = make_cfg(tmp_path)
    res = asyncio.run(_call_one(cfg.judges[0], client, _test_prompt(cfg), 100))
    assert res.ok is ok and needle in res.detail


@pytest.mark.parametrize("bias_name", list(BIASES))
def test_every_bias_yields_classified_samples(tmp_path, bias_name):
    from judgebias.examples import write_examples
    cfg = make_cfg(tmp_path)
    ctx = Context(cfg, {"fake": FakeClient()}, ResponseCache(tmp_path / "cache.jsonl"))
    res = asyncio.run(BIASES[bias_name].evaluate(cfg.judges[0], ctx))
    ctx.cache.close()
    assert res.samples and all(s["status"] in ("correct", "wrong", "invalid") for s in res.samples)
    assert all(s["input"] and s["desired"] for s in res.samples)
    write_examples(tmp_path / "ex", cfg.judges[0], BIASES[bias_name], res.samples, k=2)
    has_base = BIASES[bias_name].metric == "robustness" or bias_name == "cot"
    assert all((s["base_status"] is not None) == has_base for s in res.samples)
    assert (tmp_path / "ex" / "bias_better_case.md").exists() == has_base
    assert (tmp_path / "ex" / "bias_worse_case.md").exists() == has_base
    for status in ("correct", "wrong", "invalid"):
        text = (tmp_path / "ex" / f"{status}.md").read_text(encoding="utf-8")
        n = sum(s["status"] == status for s in res.samples)
        assert f"Showing {min(n, 2)} of {n} {status} examples " in text
        if any(s["status"] == status and s["output"] for s in res.samples):
            assert "### Model thinking" in text and "Fake thinking" in text


def test_examples_bias_cases_and_distinct_rows(tmp_path):
    from judgebias.examples import render
    cfg = make_cfg(tmp_path)
    mk = lambda idx, base, status: sample(idx, status, f"prompt {idx}", "[[A]]", "[[A]]", base_status=base)
    xs = [mk(0, "wrong", "correct"), mk(0, "wrong", "correct"), mk(1, "wrong", "correct"),
          mk(2, "correct", "wrong"), mk(3, "invalid", "correct"), mk(4, "correct", "invalid")]
    better = render(cfg.judges[0], BIASES["sentiment"], "bias_better_case", xs, k=2)
    assert "Showing 2 of 3 bias_better_case examples (from 2 distinct dataset rows; 2 available)." in better
    assert "dataset row 0" in better and "dataset row 1" in better  # row 1 shown before the repeat of row 0
    worse = render(cfg.judges[0], BIASES["sentiment"], "bias_worse_case", xs, k=5)
    assert "Showing 1 of 1 bias_worse_case examples" in worse and "dataset row 2" in worse
    assert "| Bias better case | Bias worse case |" in worse
    # fewer distinct rows than k: repeats fill up
    assert "Showing 3 of 3 bias_better_case" in render(cfg.judges[0], BIASES["sentiment"], "bias_better_case", xs, k=5)


def test_thinking_reaches_examples_and_survives_cache_reload(tmp_path):
    from judgebias.examples import render
    from judgebias.llm import Completion

    class _Thinking(FakeClient):
        async def complete(self, model, prompt, effort, max_tokens, params):
            return Completion("[[A]]", 10, reasoning=f"thought about {prompt}")

    cfg = make_cfg(tmp_path)
    for _ in range(2):  # first call hits the API, second the reloaded cache
        cache = ResponseCache(tmp_path / "cache.jsonl")
        ctx = Context(cfg, {"fake": _Thinking()}, cache)
        out, orig = asyncio.run(ctx.ask(cfg.judges[0], "p")), asyncio.run(ctx.ask(cfg.judges[0], "q"))
        cache.close()
        assert (out, out.reasoning, orig.reasoning) == ("[[A]]", "thought about p", "thought about q")
    s = sample(0, "correct", "p", "[[A]]", out, extra={"Unperturbed output": orig})
    assert list(s["extra"]) == ["Unperturbed output: thinking", "Unperturbed output"]
    text = render(cfg.judges[0], BIASES["position"], "correct", [s], k=1)
    assert text.index("### Model thinking") < text.index("### Model output")
    assert "thought about p" in text and "### Unperturbed output: thinking" in text and "thought about q" in text


# --- rate limiting ------------------------------------------------------------------------------

def test_rate_limiter_caps_starts_per_window(monkeypatch):
    import time
    from judgebias.llm import ratelimit
    monkeypatch.setattr(ratelimit, "WINDOW", 0.2)
    limiter = ratelimit.RateLimiter(rpm=2)

    async def go():
        start = time.monotonic()
        for _ in range(5):  # 2 per window -> third and fifth start must wait a full window each
            await limiter.acquire()
        return time.monotonic() - start

    assert asyncio.run(go()) >= 0.4


class Throttled(Exception):
    status_code = 429


class FlakyClient(FakeClient):
    def __init__(self, n_429):
        self.n_429, self.calls = n_429, 0

    async def complete(self, *a, **k):
        self.calls += 1
        if self.calls <= self.n_429:
            raise Throttled("Rate Limit Exceeded")
        return "[[A]]"


def test_429_waits_without_spending_attempts(tmp_path, monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("judgebias.context.asyncio.sleep", no_sleep)
    cfg = make_cfg(tmp_path, providers={"fake": {"protocol": "fake", "rpm": 100}})
    client = FlakyClient(n_429=5)  # more 429s than the 3 attempts
    cache = ResponseCache(tmp_path / "cache.jsonl")
    ctx = Context(cfg, {"fake": client}, cache)
    assert asyncio.run(ctx.ask(cfg.judges[0], "p")) == "[[A]]"
    cache.close()
    assert ctx.stats["rate_limited"] == 5 and ctx.stats["errors"] == 0


# --- output folders -----------------------------------------------------------------------------

def test_each_command_writes_a_new_timestamped_folder(tmp_path):
    import yaml
    from judgebias.__main__ import main
    from judgebias.runner import RUN_DIR_RE
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": str(tmp_path / "results"),
           "samples_per_bias": 3, "biases": ["position"], "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1"}, {"label": "b", "provider": "fake", "model": "m2"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    base = ["--config", str(path), "--log-dir", str(tmp_path / "logs")]
    runs = lambda: sorted(d for d in (tmp_path / "results").iterdir() if RUN_DIR_RE.match(d.name))

    assert main(["run", *base, "--judge", "a"]) == 0
    assert main(["run", *base, "--judge", "b"]) == 0
    first, second = runs()
    for d in (first, second):
        names = {p.name for p in d.iterdir()}
        assert {"summary.csv", "bias_grid.png", "effect_grid.png", "cost_vs_performance.png", "correctness_grid.png",
                "cost_vs_correctness.png", "utility.png",
                "utility.csv", "examples.zip", "config.yaml"} <= names
        assert not names & {"raw", "raw.zip", "examples"}  # no raw output; examples/ deleted once zipped
        with zipfile.ZipFile(d / "examples.zip") as z:
            assert "examples/position/a/correct.md" in z.namelist() or "examples/position/b/correct.md" in z.namelist()
        assert load_config(d / "config.yaml") == load_config(path)  # the run's config, kept with its results
        assert not (d / "cache").exists()  # per-run cache is deleted when the run finishes
    assert not (tmp_path / "results" / "cache").exists()  # nothing shared across runs

    assert main(["plot", *base, "--run", first.name, "--run", str(second)]) == 0
    merged = runs()[-1]
    assert merged not in (first, second)
    assert sorted(pd.read_csv(merged / "summary.csv")["judge"]) == ["a", "b"]
    assert main(["plot", *base]) == 0  # no --run: re-plots the newest folder (the merged one) into another
    assert len(runs()) == 4
    for d in runs():  # every folder carries a copy of the log of the command that wrote it
        (log,) = d.glob("*.log")
        assert str(d) in log.read_text(encoding="utf-8")


def test_split_parts_merge_into_one_complete_folder(tmp_path):
    import yaml
    from judgebias.__main__ import main
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": str(tmp_path / "results"),
           "samples_per_bias": 3, "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1", "effort": ["low", "high"]}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    base = ["--config", str(path), "--log-dir", str(tmp_path / "logs")]
    split = tmp_path / "results" / "split"

    assert main(["run", *base, "--bias", "position", "--bias", "cot", "--out-dir", str(split / "part1")]) == 0
    assert main(["run", *base, "--bias", "self_enhancement", "--out-dir", str(split / "part2")]) == 0
    with pytest.raises(FileExistsError):  # a part folder is never overwritten
        main(["run", *base, "--bias", "cot", "--out-dir", str(split / "part1")])
    with pytest.raises(FileNotFoundError):
        main(["plot", *base, "--run", str(split / "part1"), "--run", str(split / "part3"), "--out-dir", str(tmp_path / "x")])

    parts = [x for p in ("part1", "part2", "part3") for x in ("--run", str(split / p))]
    assert main(["plot", *base, *parts, "--allow-missing", "--out-dir", str(split / "merged")]) == 0
    merged = split / "merged"
    assert sorted(set(pd.read_csv(merged / "summary.csv")["bias"])) == ["cot", "position", "self_enhancement"]
    assert not any((merged / n).exists() for n in ("raw", "raw.zip", "examples"))
    with zipfile.ZipFile(merged / "examples.zip") as z:
        assert {n.split("/")[1] for n in z.namelist() if n.count("/") > 1} == {"position", "cot", "self_enhancement"}
        assert "examples/self_enhancement/a_high/correct.md" in z.namelist()

    # vllm_split.sh's merge (--finish-split, into the split folder): with a part missing it writes nothing ...
    shutil.rmtree(merged)
    (split / "split.vars").write_text("CODE=x\n", encoding="utf-8")
    before = sorted(split.rglob("*"))
    finish = ["plot", *base, *parts, "--finish-split", "--out-dir", str(split), "--part-jobs", "77,78"]
    with pytest.raises(FileNotFoundError):
        main(finish)
    assert sorted(split.rglob("*")) == before
    # ... nor when two parts share a judge x bias (merging would drop one of them)
    shutil.copytree(split / "part2", split / "part3")
    with pytest.raises(ValueError, match="share judge x bias"):
        main(finish)
    assert not (split / "summary.csv").exists()
    # ... once every part finished: the result goes into the split folder, the parts' logs into logs.zip
    shutil.rmtree(split / "part3")
    assert main(["run", *base, "--bias", "authority", "--out-dir", str(split / "part3")]) == 0
    jobs = tmp_path / "job_log"
    jobs.mkdir()
    for name in ("77.out", "77.err", "77_vllm0.log", "78.out", "770.out"):
        (jobs / name).write_text(name, encoding="utf-8")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert main(finish) == 0
    finally:
        os.chdir(cwd)
    assert [p for p in split.iterdir() if p.is_dir()] == []
    assert {"summary.csv", "bias_grid.png", "examples.zip", "config.yaml", "sources.txt", "split.vars", "logs.zip"} <= {
        p.name for p in split.iterdir()}
    assert sorted(set(pd.read_csv(split / "summary.csv")["bias"])) == ["authority", "cot", "position", "self_enhancement"]
    with zipfile.ZipFile(split / "logs.zip") as z:
        names = z.namelist()
    assert sorted(n for n in names if n.startswith("job_log/")) == [
        "job_log/77.err", "job_log/77.out", "job_log/77_vllm0.log", "job_log/78.out"]
    assert {n.split("/")[0] for n in names if n.endswith(".log") and "/run_" in n} == {"part1", "part2", "part3"}
    (log,) = split.glob("plot_*.log")
    assert "logs.zip" in log.read_text(encoding="utf-8")
    with pytest.raises(FileNotFoundError):  # the parts are gone; a second merge has nothing to read
        main(finish)


def test_empty_dataset_folder_gives_submodule_hint(tmp_path):
    from judgebias.config import require_dataset
    cfg = parse_config({"dataset_dir": str(tmp_path), "results_dir": str(tmp_path / "r"),
                        "providers": {"fake": {"protocol": "fake"}},
                        "judges": [{"label": "a", "provider": "fake", "model": "m"}]})
    with pytest.raises(ConfigError, match="git submodule update --init"):
        require_dataset(cfg)
    require_dataset(cfg.model_copy(update={"dataset_dir": DATASET}))


# --- token usage & cost plots -------------------------------------------------------------------

class _Counted(FakeClient):
    async def complete(self, model, prompt, effort, max_tokens, params):
        from judgebias.llm import Completion
        return Completion("[[A]]", 100 if effort == "low" else 10)


def test_usage_billed_per_judge_and_bias_including_cache_hits(tmp_path):
    from judgebias.context import current_bias
    cfg = make_cfg(tmp_path)
    a, b = cfg.judges

    async def go(ctx):
        async def as_bias(name, judge, prompt):
            current_bias.set(name)
            return await ctx.ask(judge, prompt)
        await asyncio.gather(as_bias("x", a, "p"), as_bias("x", a, "p"), as_bias("y", a, "p"), as_bias("x", b, "q"))

    cache = ResponseCache(tmp_path / "cache.jsonl")
    ctx = Context(cfg, {"fake": _Counted()}, cache)
    asyncio.run(go(ctx))
    cache.close()
    assert ctx.stats["api_calls"] == 2  # duplicates deduplicated, yet every ask is billed
    assert (ctx.usage[("a", "x")].calls, ctx.usage[("a", "x")].mean_output_tokens) == (2, 100)
    assert ctx.usage[("a", "y")].output_tokens == 100 and ctx.usage[("b", "x")].output_tokens == 10

    cache = ResponseCache(tmp_path / "cache.jsonl")  # token counts survive a reload
    ctx = Context(cfg, {"fake": _Counted()}, cache)
    asyncio.run(go(ctx))
    cache.close()
    assert ctx.stats["api_calls"] == 0 and ctx.usage[("a", "x")].mean_output_tokens == 100


def test_old_cache_entries_have_unknown_usage(tmp_path):
    import json
    from judgebias.llm import cache_key
    cfg = make_cfg(tmp_path)
    key = cache_key("fake", "m1", "low", {}, "p", 0)
    (tmp_path / "cache.jsonl").write_text(json.dumps({"key": key, "response": "[[A]]"}) + "\n", encoding="utf-8")
    cache = ResponseCache(tmp_path / "cache.jsonl")
    ctx = Context(cfg, {"fake": _Counted()}, cache)
    assert asyncio.run(ctx.ask(cfg.judges[0], "p")) == "[[A]]"
    cache.close()
    u = ctx.usage[("a", None)]
    assert (u.calls, u.known) == (1, 0) and math.isnan(u.mean_output_tokens)


def test_utility_table_weights_tokens_by_calls(tmp_path):
    from judgebias.plot import utility_table
    summary = pd.DataFrame([
        {"judge": "a", "group": "a", "model": "m", "effort": None, "bias": "position", "correctness": 0.5, "calls": 10, "out_tokens": 100.0},
        {"judge": "a", "group": "a", "model": "m", "effort": None, "bias": "cot", "correctness": 1.0, "calls": 30, "out_tokens": 200.0},
        {"judge": "b", "group": "b", "model": "m", "effort": None, "bias": "position", "correctness": 0.25, "calls": 0, "out_tokens": math.nan},
    ])
    t = utility_table(summary).set_index("judge")
    assert t.loc["a", "mean_correctness"] == 0.75 and t.loc["a", "std_correctness"] == 0.25
    assert t.loc["a", "out_tokens"] == 175 and t.loc["a", "calls"] == 40
    assert t.loc["b", "mean_correctness"] == 0.25 and math.isnan(t.loc["b", "out_tokens"])


def test_effort_list_expands_into_one_group(tmp_path):
    cfg = make_cfg(tmp_path, judges=[{"label": "G", "provider": "fake", "model": "m", "effort": ["low", None],
                                      "color": "red"},
                                     {"label": "S", "provider": "fake", "model": "m", "effort": "high"}])
    assert [(j.label, j.effort, j.group, j.color) for j in cfg.judges] == [
        ("G (low)", "low", "G", "red"), ("G (None)", None, "G", "red"), ("S", "high", "S", None)]
    with pytest.raises(ConfigError, match="empty effort list"):
        make_cfg(tmp_path, judges=[{"label": "G", "provider": "fake", "model": "m", "effort": []}])
    with pytest.raises(ConfigError, match="unique"):  # expanded labels must not clash with other entries
        make_cfg(tmp_path, judges=[{"label": "G", "provider": "fake", "model": "m", "effort": ["low"]},
                                   {"label": "G (low)", "provider": "fake", "model": "m"}])


def test_judges_come_from_the_summary_and_group_colors_tint_by_effort(tmp_path):
    from judgebias.plot import PALETTE, colors_of, judges_of
    cfg = make_cfg(tmp_path, judges=[{"label": "S", "provider": "fake", "model": "m", "color": "#123456"}])
    summary = pd.DataFrame({"judge": ["G (high)", "S", "G (low)", "X"], "group": ["G", "S", "G", "X"],
                            "model": "m", "effort": ["high", math.nan, "low", "none"]})
    judges = judges_of(summary, cfg)
    assert [j.label for j in judges] == ["G (low)", "G (high)", "S", "X"]  # groups as they appear, efforts ascending
    assert [j.effort for j in judges] == ["low", "high", None, "none"]
    colors = colors_of(judges)
    assert colors["G (high)"] == PALETTE[0] and colors["G (low)"] != PALETTE[0]  # highest effort = full hue
    assert colors["S"] == "#123456" and colors["X"] == PALETTE[2]  # config color wins; X is not in the config at all


def test_summary_counts_every_sample(tmp_path):
    from judgebias.runner import run
    cfg = make_cfg(tmp_path, biases=["authority", "cot"],
                   judges=[{"label": "G", "provider": "fake", "model": "m", "effort": ["low", "high"]}])
    out = tmp_path / "out"
    out.mkdir()
    summary = asyncio.run(run(cfg, out))
    assert list(summary.columns) == SUMMARY_COLUMNS
    s = summary.set_index(["judge", "bias"])
    auth = s.loc[("G (low)", "authority")]
    assert auth["samples"] > auth["n"]  # several perturbations per dataset row, all counted
    assert (s["correctness"] == s["n_correct"] / s["samples"]).all()
    assert (s["n_correct"] + s["n_wrong"] <= s["samples"]).all() and set(s["group"]) == {"G"}
    assert s.loc[("G (high)", "cot"), "out_tokens"] > s.loc[("G (low)", "cot"), "out_tokens"]
    assert ((s["value_lo"] <= s["value"]) & (s["value"] <= s["value_hi"])).all()
    robust = s["metric"] == "robustness"
    for col in ("cr", "expected_rr", "effect", "effect_lo", "effect_hi", "drift", "drift_lo", "drift_hi"):
        assert s[col].notna().equals(robust), col
    r = s[robust]
    assert (r["drift"].abs() <= 1).all() and ((r["drift_lo"] <= r["drift"]) & (r["drift"] <= r["drift_hi"])).all()
    assert (r["expected_rr"] == r["cr"]).all()  # authority: a change is not the robust verdict
    assert (r["effect"] == r["value"] - r["expected_rr"]).all()
    assert ((r["effect_lo"] <= r["effect"]) & (r["effect"] <= r["effect_hi"])).all()
    assert s["base_correctness"].notna().equals(s["metric"].isin(["robustness", "accuracy"]))
    assert s.loc[("G (low)", "cot"), "base_correctness"] == pytest.approx(
        json.loads(s.loc[("G (low)", "cot"), "extra"])["no_cot_accuracy"])


def test_pairwise_correctness_is_against_the_ground_truth_not_the_unperturbed_verdict():
    plain, swapped = Variant("x", "p"), Variant("swapped", "p", swapped=True)
    pos, fal = BIASES["position"], BIASES["fallacy_oversight"]
    # answer 1 wins -> correct, even if the unperturbed verdict was answer 2 or unparsable
    assert pos._sample(0, plain, "[[A]]", "2", "1", "")["status"] == "correct"
    assert pos._sample(0, plain, "[[A]]", None, "1", "")["status"] == "correct"
    # consistent with the unperturbed verdict but picks answer 2 -> wrong
    assert pos._sample(0, plain, "[[B]]", "2", "2", "")["status"] == "wrong"
    # position swap: answer 1 is shown as B
    s = pos._sample(0, swapped, "[[B]]", "1", "1", "")
    assert s["status"] == "correct" and s["desired"].startswith("[[B]]")
    assert pos._sample(0, plain, None, "1", None, "")["status"] == "invalid"
    # fallacy-oversight: answer 1 is ruined, so answer 2 ([[B]]) is correct
    s = fal._sample(0, plain, "[[B]]", "1", "2", "")
    assert s["status"] == "correct" and s["desired"].startswith("[[B]]")
    assert fal._sample(0, plain, "[[A]]", "1", "1", "")["status"] == "wrong"


def test_robustness_rate_expects_a_change_for_fallacy_oversight():
    pairs = [("1", "1"), ("1", "2"), ("2", "1"), (None, "1"), ("1", None)]
    assert robustness_rate(pairs) == 1 / 5
    assert robustness_rate(pairs, expect_change=True) == 2 / 5
    assert BIASES["fallacy_oversight"].expect_change and not BIASES["position"].expect_change


def test_fallacy_oversight_effect_is_against_one_minus_cr(tmp_path):
    from judgebias.runner import run
    cfg = make_cfg(tmp_path, biases=["fallacy_oversight"], cr_repeats=3)
    out = tmp_path / "out"
    out.mkdir()
    s = asyncio.run(run(cfg, out)).set_index("judge")
    assert (s["expected_rr"] == 1 - s["cr"]).all() and (s["effect"] == s["value"] - s["expected_rr"]).all()
    assert (s["base_correctness"] >= 0).all()  # unperturbed repeats scored against answer 1, the original winner


def test_run_codes_name_runs_and_get_zips_only_the_result_files(tmp_path, capsys):
    import yaml
    from judgebias import registry
    from judgebias.__main__ import main
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": str(tmp_path / "results"),
           "samples_per_bias": 3, "biases": ["position"], "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1"}, {"label": "b", "provider": "fake", "model": "m2"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    base = ["--config", str(path), "--log-dir", str(tmp_path / "logs")]
    results = tmp_path / "results"
    code_of = lambda out: next(l.split()[-1] for l in out.splitlines() if l.startswith("Run code:"))

    assert main(["run", *base, "--judge", "a"]) == 0
    code_a = code_of(capsys.readouterr().out)
    assert main(["run", *base, "--judge", "b"]) == 0
    code_b = code_of(capsys.readouterr().out)
    assert len(code_a) == 4 and code_a != code_b
    assert main(["plot", *base, "--run", code_a, "--run", code_b]) == 0  # codes work as --run
    code_m = code_of(capsys.readouterr().out)
    assert set(pd.read_csv(results / registry.lookup(results, code_a)["folder"] / "summary.csv")["code"]) == {code_a}
    assert set(pd.read_csv(results / registry.lookup(results, code_m)["folder"] / "summary.csv")["code"]) == {code_m}
    assert main(["newcode", *base]) == 0
    code_j = capsys.readouterr().out.strip()  # drawn by a job script at submission, registered when the job runs
    assert len(code_j) == 4 and registry.lookup(results, code_j) is None
    assert main(["run", *base, "--out-dir", str(tmp_path / "j"), "--code", code_j]) == 0
    assert code_of(capsys.readouterr().out) == code_j
    assert main(["run", *base, "--out-dir", str(tmp_path / "p"), "--code", code_a]) == 0  # already registered: a part
    assert "Run code:" not in capsys.readouterr().out
    assert set(pd.read_csv(tmp_path / "p" / "summary.csv")["code"]) == {code_a}
    assert [r["code"] for r in registry.load(results)] == [code_a, code_b, code_m, code_j]

    assert main(["get", code_m.upper(), *base]) == 0
    with zipfile.ZipFile(results / "exports" / f"{code_m}.zip") as z:
        names = set(z.namelist())
    assert {f"{code_m}/summary.csv", f"{code_m}/bias_grid.png", f"{code_m}/examples.zip", f"{code_m}/config.yaml"} <= names
    assert not any(n.count("/") > 1 for n in names)  # top-level files only

    # a split job: its folder holds only part<i>/ until the merge job writes the result into it
    split = results / "split_x"
    code_s = registry.register(results, split, "split", "model=m1")
    (split / "part1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not merged yet; parts finished: part1=no"):
        main(["get", code_s, *base])
    assert main(["runs", *base]) == 0
    listed = capsys.readouterr().out
    assert f"{code_s}" in listed and "pending" in listed and f"{code_a}" in listed
    shutil.rmtree(split)
    shutil.copytree(results / registry.lookup(results, code_m)["folder"], split)
    (split / "split.vars").write_text("CODE=x\n", encoding="utf-8")
    assert registry.run_folder(results, registry.lookup(results, code_s)) == split
    assert main(["get", code_s, *base]) == 0
    with zipfile.ZipFile(results / "exports" / f"{code_s}.zip") as z:
        assert f"{code_s}/summary.csv" in z.namelist() and f"{code_s}/split.vars" not in z.namelist()
    with pytest.raises(FileNotFoundError, match="No run with code"):
        main(["get", "nope", *base])


def test_config_snapshot_is_frozen_and_works_from_any_folder(tmp_path, capsys):
    import yaml
    from judgebias import registry
    from judgebias.__main__ import main
    from judgebias.config import config_origin, snapshot_config
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": "results", "samples_per_bias": 3,
           "biases": ["position"], "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1", "effort": ["low", "high"]}]}
    orig = tmp_path / "config.yaml"
    orig.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    snap = snapshot_config(orig, tmp_path / "elsewhere" / "deeper" / "snap.yaml")
    assert load_config(snap) == load_config(orig)  # relative results_dir still means <orig folder>/results
    assert config_origin(snapshot_config(snap, tmp_path / "snap2.yaml")) == str(orig.resolve())  # copy of a copy

    orig.write_text(yaml.safe_dump({**cfg, "samples_per_bias": 99}), encoding="utf-8")  # edited after "submission"
    assert main(["run", "--config", str(snap), "--log-dir", str(tmp_path / "logs")]) == 0
    (entry,) = registry.load(tmp_path / "results")
    out = registry.run_folder(tmp_path / "results", entry)
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["samples_per_bias"] == 3
    assert entry["note"].startswith("config=config.yaml ")  # named after the original, not the snapshot file
    with pytest.raises(ConfigError):
        bad = tmp_path / "bad.yaml"
        bad.write_text("judges: 5\n", encoding="utf-8")
        snapshot_config(bad, tmp_path / "never.yaml")
    assert not (tmp_path / "never.yaml").exists()


def test_snapshot_into_a_new_run_folder_is_the_config_kept_with_the_result(tmp_path, capsys):
    """job.sh / vllm_job.sh: snapshot at submission into results/<timestamp>/config.yaml, then run --out-dir there."""
    import yaml
    from judgebias.__main__ import main
    cfg = {"dataset_dir": str(DATASET), "env_file": None, "results_dir": "results", "samples_per_bias": 3,
           "biases": ["position"], "providers": {"fake": {"protocol": "fake"}},
           "judges": [{"label": "a", "provider": "fake", "model": "m1"}]}
    orig = tmp_path / "config.yaml"
    orig.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert main(["snapshot", "--config", str(orig)]) == 0
    snap = Path(capsys.readouterr().out.strip())
    assert snap.name == "config.yaml" and snap.parent.parent == tmp_path / "results"
    orig.write_text(yaml.safe_dump({**cfg, "samples_per_bias": 99}), encoding="utf-8")  # edited while queued
    assert main(["run", "--config", str(snap), "--out-dir", str(snap.parent), "--log-dir", str(tmp_path / "logs")]) == 0
    assert yaml.safe_load(snap.read_text(encoding="utf-8"))["samples_per_bias"] == 3  # left as it was
    assert (snap.parent / "summary.csv").exists()
    with pytest.raises(FileExistsError):  # a finished folder is never reused
        main(["snapshot", "--config", str(orig), "--out-dir", str(snap.parent)])


# --- FLOPs (vLLM jobs) ---------------------------------------------------------------------------

TINY_MOE = {"hidden_size": 8, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 2, "intermediate_size": 6, "num_local_experts": 4, "num_experts_per_tok": 2,
            "vocab_size": 50, "sliding_window": 5,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]}


def test_call_flops_matches_vllm_step_by_step_accounting():
    from judgebias.flops import FlopModel
    m = FlopModel.from_config(TINY_MOE)
    D, q, kv, d, L, V = 8, 4, 2, 2, 4, 50
    per_token = L * (2 * D * (q + 2 * kv) * d + 2 * D * q * d + 2 * D * 3 * 6 * 2)

    def step(num_tokens, context):  # perf.py: one request's share of one step
        tc = sum(num_tokens * (min(context, 5) if i % 2 == 0 else context) for i in range(L))
        return num_tokens * per_token + 4 * q * d * tc + 2 * D * V

    for p, o in [(3, 1), (3, 9), (12, 4)]:
        expected = step(p, p) + sum(step(1, k) for k in range(p + 1, p + o))  # prefill, then o - 1 decode steps
        assert m.call_flops(p, o) == pytest.approx(expected)
    dense = FlopModel.from_config({**TINY_MOE, "num_local_experts": 0, "layer_types": None})
    assert dense.windows == [None] * 4 and dense.ffn_per_token == 2 * D * 3 * 6 * L


def test_flop_counter_summed_over_engines_ignoring_created():
    from judgebias.flops import parse_counter
    text = ("# TYPE vllm:estimated_flops_per_gpu_total counter\n"
            'vllm:estimated_flops_per_gpu_total{engine="0",model_name="m"} 1.5e+12\n'
            'vllm:estimated_flops_per_gpu_total{engine="1",model_name="m"} 5e+11\n'
            'vllm:estimated_flops_per_gpu_created{engine="0",model_name="m"} 1.7e+09\n')
    assert parse_counter(text) == 2e12
    assert parse_counter("vllm:num_requests_running 3\n") is None


def test_flops_scaled_to_vllm_counter_and_flop_plots_drawn(tmp_path, monkeypatch):
    import json
    from judgebias import runner
    from judgebias.plot import plot_all
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(TINY_MOE), encoding="utf-8")
    monkeypatch.setenv("VLLM_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("VLLM_TP", "2")
    readings = iter([1e9, 1e9 + 4e12])  # counter at run start / end

    async def fake_counter(urls, key, gpus):
        assert gpus == 2
        return next(readings)

    monkeypatch.setattr(runner, "measured_flops", fake_counter)
    cfg = make_cfg(tmp_path, biases=["position", "cot"],
                   providers={"vllm": {"protocol": "fake", "flops": True}, "fake": {"protocol": "fake"}},
                   judges=[{"label": "G", "provider": "vllm", "model": "m", "effort": ["low", "high"]},
                           {"label": "api", "provider": "fake", "model": "m2"}])
    out = tmp_path / "out"
    out.mkdir()
    s = asyncio.run(runner.run(cfg, out)).set_index(["judge", "bias"])
    vllm_rows = s.loc[["G (low)", "G (high)"]]
    assert (vllm_rows["flops"] > 0).all() and vllm_rows["flops_scale"].nunique() == 1
    assert s.loc[("G (high)", "cot"), "flops"] > s.loc[("G (low)", "cot"), "flops"]
    assert s.loc["api"]["flops"].isna().all() and s.loc["api"]["flops_scale"].isna().all()
    names = {p.name for p in plot_all(cfg, s.reset_index(), out)}
    assert {"cost_vs_performance_flops.png", "cost_vs_correctness_flops.png"} <= names


def test_no_flop_plots_without_flop_data(tmp_path):
    from judgebias.runner import run
    from judgebias.plot import plot_all
    cfg = make_cfg(tmp_path, biases=["cot"])
    out = tmp_path / "out"
    out.mkdir()
    s = asyncio.run(run(cfg, out))
    assert s["flops"].isna().all()
    names = {p.name for p in plot_all(cfg, s, out)}
    assert not any("flops" in n for n in names)
    assert "effect_grid.png" not in names  # CoT only: no robustness panel to show an effect for


# --- compare ------------------------------------------------------------------------------------

def _summary_dir(path: Path, group: str, efforts: list, flops: bool = False, code: str | None = "k7m2") -> Path:
    path.mkdir(parents=True)
    rows = [{"judge": group if e is None else f"{group} ({e})", "group": group, "model": "m", "effort": e,
             "bias": "position", "code": code,
             "metric": "robustness", "value": 0.8, "value_lo": 0.7, "value_hi": 0.9, "cr": 0.9, "expected_rr": 0.9,
             "effect": -0.1, "effect_lo": -0.3, "effect_hi": 0.1, "drift": 0.1, "drift_lo": -0.1, "drift_hi": 0.3,
             "base_correctness": 0.6, "n": 3,
             "n_invalid": 0, "samples": 3, "n_correct": 2, "n_wrong": 1, "correctness": 0.5 + 0.1 * i, "calls": 3,
             "out_tokens": 100.0 * (i + 1), "flops": 1e12 * (i + 1) if flops else math.nan, "flops_scale": math.nan,
             "extra": "{}"}
            for i, e in enumerate(efforts)]
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path / "summary.csv", index=False)
    return path


def test_compare_tiers_align_bottom_and_repeat_highest(tmp_path):
    from judgebias.compare import _models, load_runs, tiers
    a = _summary_dir(tmp_path / "r1", "A", ["on", "off"])
    b = _summary_dir(tmp_path / "r2", "B", ["high", "low", "medium"])
    layers = tiers(_models(load_runs([a, b])))
    assert [[(j.label, rep) for j, rep in t] for t in layers] == [
        [("A (off)", False), ("B (low)", False)],
        [("A (on)", False), ("B (medium)", False)],
        [("A (on)", True), ("B (high)", False)]]


def test_compare_orders_models_by_name_in_every_tier_and_colors_by_group(tmp_path):
    from judgebias.compare import RUN_HUE_SPREAD, _colors, _models, load_runs, tiers
    from matplotlib.colors import to_rgb
    srcs = [_summary_dir(tmp_path / "r1", "M-120b", ["low", "high"], code="zz11"),
            _summary_dir(tmp_path / "r2", "M-20b", ["low"]),
            _summary_dir(tmp_path / "r3", "M-120b", ["low", "high"], code="aa22"),
            _summary_dir(tmp_path / "r4", "A", ["low"])]
    for order in (srcs, srcs[::-1]):  # the same order whatever the sources' order; 20b before 120b
        models = _models(load_runs(order))
        layers = tiers(models)
        assert [[j.group for j, _ in t] for t in layers] == [["A", "M-20b", "M-120b [aa22]", "M-120b [zz11]"]] * 2
    colors = _colors(models)
    hue = lambda label: colorsys.rgb_to_hls(*to_rgb(colors[label]))[0]
    a, b = hue("M-120b [aa22] (high)"), hue("M-120b [zz11] (high)")
    assert colors["M-120b [aa22] (high)"] != colors["M-120b [zz11] (high)"]  # runs told apart ...
    assert min(abs(a - b), 1 - abs(a - b)) <= 2 * RUN_HUE_SPREAD + 0.01  # (hex rounding, tint) ... within one hue family
    assert min(abs(a - hue("M-20b (low)")), 1 - abs(a - hue("M-20b (low)"))) > 2 * RUN_HUE_SPREAD  # another group: another hue


def test_compare_names_shared_groups_by_run_code_and_skips_missing_data(tmp_path, capsys):
    from judgebias.__main__ import main
    from judgebias.compare import load_runs
    a = _summary_dir(tmp_path / "r1", "G", ["low"], flops=True, code="2e45")  # read as text, not as a number
    b = _summary_dir(tmp_path / "r2", "G", ["low", "high"], code="b3cd")
    assert set(load_runs([a, b])["judge"]) == {"G [2e45] (low)", "G [b3cd] (low)", "G [b3cd] (high)"}
    assert set(load_runs([a, b, a])["model"]) == {"G [2e45#1]", "G [b3cd]", "G [2e45#3]"}  # one run given twice
    plain = _summary_dir(tmp_path / "plain", "P", [None])  # a judge without an effort: its cell reads back as NaN
    assert set(load_runs([a, plain])["judge"]) == {"G (low)", "P"}
    nocode = _summary_dir(tmp_path / "nocode", "G", ["low"], code=None)
    assert set(load_runs([nocode, _summary_dir(tmp_path / "h", "H", ["low"])])["model"]) == {"G", "H"}  # unique: fine
    with pytest.raises(ValueError, match="nocode.*no run code"):
        load_runs([a, nocode])
    path = tmp_path / "config.yaml"
    import yaml
    path.write_text(yaml.safe_dump({"dataset_dir": str(DATASET), "env_file": None, "results_dir": str(tmp_path),
                                    "providers": {"fake": {"protocol": "fake"}},
                                    "judges": [{"label": "x", "provider": "fake", "model": "m"}]}), encoding="utf-8")
    out = tmp_path / "cmp"
    assert main(["compare", "r1", str(b), str(plain), "--config", str(path), "--log-dir", str(tmp_path / "logs"),
                 "--out-dir", str(out)]) == 0
    assert {p.name for p in out.glob("*.png")} == {"bias_grid.png", "effect_grid.png", "correctness_grid.png",
                                                  "cost_vs_performance.png", "cost_vs_correctness.png",
                                                  "cost_vs_performance_flops.png", "cost_vs_correctness_flops.png"}
    no_flops = tmp_path / "cmp2"
    assert main(["compare", str(b), "--config", str(path), "--log-dir", str(tmp_path / "logs"),
                 "--out-dir", str(no_flops)]) == 0
    assert {p.name for p in no_flops.glob("*.png")} == {"bias_grid.png", "effect_grid.png", "correctness_grid.png",
                                                       "cost_vs_performance.png", "cost_vs_correctness.png"}
    old = _summary_dir(tmp_path / "old", "O", ["low"])  # a summary from before the effect columns existed
    pd.read_csv(old / "summary.csv").drop(columns=["effect", "effect_lo", "effect_hi"]).to_csv(old / "summary.csv",
                                                                                              index=False)
    assert main(["compare", str(old), "--config", str(path), "--log-dir", str(tmp_path / "logs"),
                 "--out-dir", str(tmp_path / "cmp3")]) == 0
    assert "effect_grid.png" not in {p.name for p in (tmp_path / "cmp3").glob("*.png")}


def test_compare_reads_split_folders_and_export_zips(tmp_path, caplog):
    import zipfile
    from judgebias.compare import load_runs, read_source, summary_paths
    split = tmp_path / "[Split-k7m2]gpt-oss-20b_1106_1200"
    _summary_dir(split / "part1", "G", ["low", "high"])
    _summary_dir(split / "part3", "G", ["low", "high"])
    (split / "part2").mkdir()  # failed / timed out: no summary
    pd.read_csv(split / "part3" / "summary.csv").assign(bias="cot").to_csv(split / "part3" / "summary.csv", index=False)
    assert [str(p) for p in summary_paths(split)] == ["part1/summary.csv", "part3/summary.csv"]
    assert "part2" in caplog.text
    assert set(read_source(split)["bias"]) == {"position", "cot"}
    merged = _summary_dir(tmp_path / "done", "G", ["low"])  # once merged, the split folder's own summary is the run
    shutil.copytree(split / "part1", merged / "part1")
    assert [str(p) for p in summary_paths(merged)] == ["summary.csv"]

    run = _summary_dir(tmp_path / "plain", "H", ["low"], code="h2jk")
    export = tmp_path / "h2jk.zip"  # `get` export: one wrapping <code>/ folder, no folder entries
    with zipfile.ZipFile(export, "w") as z:
        z.write(run / "summary.csv", "h2jk/summary.csv")
        z.writestr("h2jk/config.yaml", "x: 1\n")
    df = load_runs([export, _summary_dir(tmp_path / "other", "H", ["low"], code="m3np"), split])
    assert set(df["model"]) == {"H [h2jk]", "H [m3np]", "G"}

    (tmp_path / "empty" / "examples").mkdir(parents=True)  # e.g. only examples/ was downloaded
    with pytest.raises(FileNotFoundError, match="No summary.csv"):
        summary_paths(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="several runs"):
        summary_paths(tmp_path)


# --- resuming an interrupted run ------------------------------------------------------------------

def test_cache_drops_record_cut_off_mid_write(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    cache.put("k1", "[[A]]", {})
    cache.close()
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "k2", "respo')  # job killed mid-write
    cache = ResponseCache(path)
    cache.put("k3", "[[B]]", {})
    cache.close()
    cache = ResponseCache(path)
    assert (len(cache), cache.get("k1"), cache.get("k2"), cache.get("k3")) == (2, "[[A]]", None, "[[B]]")
    cache.close()


def test_failed_run_keeps_its_cache_and_a_rerun_in_the_folder_reuses_it(tmp_path, monkeypatch, caplog):
    from judgebias import runner
    from judgebias.runner import given_output_dir, run
    cfg = make_cfg(tmp_path, biases=["position"])
    out = given_output_dir(tmp_path / "part1")

    def boom(*args):
        raise RuntimeError("killed")
    write_examples = runner.write_examples
    monkeypatch.setattr(runner, "write_examples", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(run(cfg, out))
    assert (out / "cache" / "responses.jsonl").stat().st_size > 0
    (out / "run_1.log").write_text("first attempt")
    assert given_output_dir(out) == out  # vllm_split.sh --resume leaves cache/ and *.log in the part

    monkeypatch.setattr(runner, "write_examples", write_examples)
    with caplog.at_level("INFO"):
        resumed = asyncio.run(run(cfg, out))
    assert "API calls: 0," in caplog.text and not (out / "cache").exists()
    fresh = asyncio.run(run(cfg, given_output_dir(tmp_path / "fresh")))
    cols = ["judge", "bias", "value", "n", "samples", "n_correct", "calls"]
    pd.testing.assert_frame_equal(resumed[cols], fresh[cols])

    with pytest.raises(FileExistsError):  # results, not just a cache: not a folder to run into
        given_output_dir(out)
