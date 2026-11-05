import asyncio
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from judgebias.biases import BIASES
from judgebias.biases.pairwise import Variant, base_prompt, to_answer
from judgebias.config import Config, ConfigError, load_config, parse_config, resolve_keys
from judgebias.context import Context
from judgebias.llm import ResponseCache, build_clients
from judgebias.llm.anthropic_proto import AnthropicProtocolClient
from judgebias.llm.fake import FakeClient
from judgebias.llm.openai_proto import OpenAIProtocolClient
from judgebias.metrics import accuracy, consistency_rate, error_rate, robustness_rate
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
    assert consistency_rate([["1", "1"], ["1", "2"], [None, None]]) == pytest.approx(1 / 3)
    assert accuracy(["1", "2", None, "1"], "1") == 0.5
    assert error_rate([(8, 8), (6, 8), (None, 8)]) == pytest.approx(0.125)
    assert math.isnan(robustness_rate([]))


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


def test_repo_configs_load():
    for name in ("config.yaml", "config.fake.yaml"):
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
    for status in ("correct", "wrong", "invalid"):
        text = (tmp_path / "ex" / f"{status}.md").read_text(encoding="utf-8")
        n = sum(s["status"] == status for s in res.samples)
        assert f"Showing {min(n, 2)} of {n} {status} examples." in text


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
        assert {"summary.csv", "bias_grid.png", "cost_vs_performance.png", "correctness_grid.png", "utility.png",
                "utility.csv", "raw", "examples", "raw.zip", "examples.zip"} <= {p.name for p in d.iterdir()}
        for name in ("raw", "examples"):  # each zip holds exactly the folder's files, byte for byte
            with zipfile.ZipFile(d / f"{name}.zip") as z:
                files = {f.relative_to(d).as_posix(): f.read_bytes() for f in (d / name).rglob("*") if f.is_file()}
                assert files and {n: z.read(n) for n in z.namelist() if not n.endswith("/")} == files
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
    cfg = make_cfg(tmp_path)
    summary = pd.DataFrame([
        {"judge": "a", "bias": "position", "correctness": 0.5, "calls": 10, "out_tokens": 100.0},
        {"judge": "a", "bias": "cot", "correctness": 1.0, "calls": 30, "out_tokens": 200.0},
        {"judge": "b", "bias": "position", "correctness": 0.25, "calls": 0, "out_tokens": math.nan},
    ])
    t = utility_table(cfg, summary).set_index("judge")
    assert t.loc["a", "mean_correctness"] == 0.75 and t.loc["a", "std_correctness"] == 0.25
    assert t.loc["a", "out_tokens"] == 175 and t.loc["a", "calls"] == 40
    assert t.loc["b", "mean_correctness"] == 0.25 and math.isnan(t.loc["b", "out_tokens"])


def test_plot_all_handles_summaries_without_cost_columns(tmp_path):
    from judgebias.plot import plot_all
    cfg = make_cfg(tmp_path)
    old = pd.DataFrame([{"judge": j, "model": "m", "effort": None, "bias": "position", "metric": "robustness",
                         "value": 0.8, "cr": 0.9, "n": 3, "n_invalid": 0, "extra": "{}"} for j in ("a", "b")])
    outs = plot_all(cfg, old, tmp_path)
    assert all(p.exists() for p in outs) and (tmp_path / "utility.csv").exists()


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


def test_group_colors_tint_by_effort(tmp_path):
    from judgebias.plot import PALETTE, _judges_and_colors
    cfg = make_cfg(tmp_path, judges=[{"label": "G", "provider": "fake", "model": "m", "effort": ["high", "low"]},
                                     {"label": "S", "provider": "fake", "model": "m"}])
    _, colors = _judges_and_colors(cfg, pd.DataFrame({"judge": ["G (high)", "G (low)", "S"]}))
    assert colors["G (high)"] == PALETTE[0] and colors["S"] == PALETTE[1]  # highest effort = full hue
    assert colors["G (low)"] != PALETTE[0]


def test_summary_counts_every_sample(tmp_path):
    from judgebias.runner import run
    cfg = make_cfg(tmp_path, biases=["authority", "cot"],
                   judges=[{"label": "G", "provider": "fake", "model": "m", "effort": ["low", "high"]}])
    out = tmp_path / "out"
    out.mkdir()
    s = asyncio.run(run(cfg, out)).set_index(["judge", "bias"])
    auth = s.loc[("G (low)", "authority")]
    assert auth["samples"] > auth["n"]  # several perturbations per dataset row, all counted
    assert (s["correctness"] == s["n_correct"] / s["samples"]).all()
    assert (s["n_correct"] + s["n_wrong"] <= s["samples"]).all() and set(s["group"]) == {"G"}
    assert s.loc[("G (high)", "cot"), "out_tokens"] > s.loc[("G (low)", "cot"), "out_tokens"]


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
