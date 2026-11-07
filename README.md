# reasoning-judge-bias-mini

A small, config-driven reproduction of the bias benchmark from
[*Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*](https://arxiv.org/abs/2410.02736) (CALM).
You list the judges you want — each a **(provider, model, thinking effort)** triple — and the tool runs the
paper's 12 bias tests on only those judges and draws a 3×4 grid like `Goal.png`.

It reuses the paper's datasets and prompt builders from `LLM-Judge-Bias-Dataset/` (not copied).

## Setup

```bash
git clone --recursive <this repo>   # LLM-Judge-Bias-Dataset/ is a git submodule
git submodule update --init --remote   # already cloned without --recursive (empty dataset folder), or to update it
uv sync                      # Python >= 3.12
cp .env.example .env         # then fill in the keys you need
```

`.env` holds secrets; `config.yaml` only holds the **name** of the variable each provider uses
(`api_key_env`), so keys never appear in the config.

## Usage

```bash
uv run python -m judgebias check --config config.yaml   # validate config + keys, then one test call per judge
uv run python -m judgebias run   --config config.yaml   # run judges x biases -> results/<timestamp>/summary.csv + figures
uv run python -m judgebias plot  --config config.yaml   # redraw the newest run into a new results/<timestamp>/
uv run python -m judgebias plot  --config config.yaml --run 20261103_150000 --run 20261103_160000  # merge runs
uv run python -m judgebias compare --config config.yaml GPT-20b-n3 Qwen27-122  # compare finished runs, no model calls
uv run pytest                                           # tests (no API calls)
```

- `run` accepts `--judge "<label>"` and `--bias <name>` (both repeatable) to run a subset. Partial runs are merged into `summary.csv`.
- `check --no-call` skips the live test calls.
- `compare RUN [RUN ...]` (run codes or folders, like `plot --run`) draws finished runs side by side into a new
  folder, without calling any model: `bias_grid.png` and `correctness_grid.png` cluster the bars by effort tier, and
  `cost_vs_performance.png` / `cost_vs_correctness.png` (and their `_flops` versions, for judges with FLOPs) overlay
  all judges. Each (folder, group) is one model (named
  `group [folder]` when two folders share a group). Tiers are bottom-aligned: each model's efforts, low to high, fill
  T1, T2, … and a model with fewer efforts repeats its highest one upward (hatched). E.g. A `off/on` with B
  `low/medium/high` gives T1 = A off, B low; T2 = A on, B medium; T3 = A on (repeated), B high. A model whose
  summary lacks the needed columns (older runs) is left out of that figure.
- Offline dry run with fake judges (no keys, no cost): `uv run python -m judgebias run --config config.fake.yaml`.

### `check`

1. Validates the config and that every needed key exists in `.env` (errors exit 2, before any call).
2. Warns about setups that give an incomplete chart (e.g. Self-Enhancement needs ≥ 2 judges).
3. Sends each judge one tiny judging prompt (uncached) and reports `OK` / `FAIL` with the error type and
   message — bad key, no credit, wrong model ID, rejected param, refusal, or reply with no `[[A]]`/`[[B]]` verdict.
   Exits 0 only if all judges pass.

Run `check` first, and start `run` with a small `samples_per_bias` (e.g. 5) as a smoke test.

## Configuration (`config.yaml`)

| Key | Meaning |
|---|---|
| `dataset_dir` | Path to the dataset repo (default `LLM-Judge-Bias-Dataset`) |
| `samples_per_bias` / `seed` | Seed-fixed subset size per dataset |
| `concurrency` | Max simultaneous API calls per model |
| `providers.<name>.rpm` | Optional client-side cap on requests per 60 s, shared by all judges on that provider (API key) |
| `cr_repeats` | Times the unperturbed prompt is judged, for Consistency Rate (≥ 2) |
| `max_tokens` | Per-response cap (thinking + answer) |
| `log_dir` | Where log files go (default `<results_dir>/logs`); override per run with `--log-dir` |
| `providers` | `name: {protocol: openai\|anthropic, base_url, api_key_env}`; any OpenAI-compatible endpoint works (use `token_param: max_tokens` if it needs it) |
| `judges` | `{label, provider, model, effort, params?, color?}` — only these are ever called; `effort` may be a list |
| `biases` | `all` or a list of bias names |
| `bias_options` | Per-bias knobs (bandwagon `number`, diversity `groups`, compassion-fade `model_names`, …) |

`effort` maps to `reasoning_effort` (OpenAI protocol) or `output_config.effort` (Anthropic protocol);
`null` sends nothing. `params` is passed through verbatim (e.g. `thinking: {type: between_tools}` to turn
Sonnet 5.5 thinking off). Judge labels must be unique.

One entry with an effort list stands for one judge per effort, labelled `"<label> (<effort>)"`, and forms one
**group**: the group shares a hue in every figure (lower effort = lighter tint), and its judges are joined by a line
in the cost plots.

```yaml
judges:
  - {label: "GPT-oss-20b", provider: Alliance, model: gpt-oss-20b, effort: [low, medium, high]}
  # = three judges "GPT-oss-20b (low)", "GPT-oss-20b (medium)", "GPT-oss-20b (high)"
```

## The 12 panels

| Panel | Data | Perturbation | Metric |
|---|---|---|---|
| Position | alignment | swap answer order | Robustness Rate |
| Verbosity | verbosity | lengthen answer 2 | RR |
| Compassion-Fade | alignment | real model names instead of "Assistant A/B" | RR |
| Bandwagon | alignment | "90% of people believe B is better" | RR |
| Distraction | alignment | irrelevant remark about assistant B | RR |
| Fallacy-Oversight | fallacy_oversight | answer 1 gets flawed reasoning | RR |
| Authority | authority | answer 2 gets a fake book/quote/URL reference (avg) | RR |
| Sentiment | sentiment | answer 1 rewritten cheerful/sad/angry/fearful (avg) | RR |
| Diversity | alignment | "Assistant B identifies as <group>" (avg) | RR |
| CoT | fact_related | CoT judging prompt | Accuracy |
| Self-Enhancement | alignment | judge answers, all judges score it | Error rate = \|self − others\| / others |
| Refinement-Aware | refinement | score with vs. without "this was refined" note | Error rate = \|Δ\| / unaware |

RR = share of items whose verdict (mapped back to the underlying answer) is unchanged by the perturbation;
unparseable verdicts count as non-robust and are tallied in `n_invalid`. Fallacy-Oversight is the exception (paper
App. C): answer 1's reasoning is made chaotic, so an item is robust when the verdict *changes*. The red dashed line is the mean
Consistency Rate across judges for RR panels, and the mean metric across judges for the others.
Perturbations that target one answer always target the rejected one (answer 2), except Fallacy-Oversight and
Sentiment (answer 1). Dataset rows missing a perturbed field are skipped, not invented.

## Layout

```
config.yaml / config.fake.yaml   real and offline configs
judgebias/
  config.py       config models, .env loading, key matching
  check.py        pre-flight check with live test calls
  llm/            protocol adapters (openai, anthropic, fake), per-run JSONL response cache
  context.py      cached, concurrency-limited `ask` shared by all biases
  data.py, prompts.py, parse.py, metrics.py
  biases/         one module/registration per panel; registry is `BIASES`
  runner.py       judge x bias loop -> results/<timestamp>/
  plot.py         3x4 bias grid, cost vs. metric / Correctness grids, utility plot
tests/            pytest suite (no network)
```

Every command also writes all log records (INFO and above, including warnings, errors and tracebacks) to
`<log_dir>/<command>_<YYYYmmdd_HHMMSS>.log`, one file per invocation; `run` and `plot` also copy it into their
`results/<YYYYmmdd_HHMMSS>/` folder when they finish (or fail).

Every `run` and `plot` writes to a new `results/<YYYYmmdd_HHMMSS>/` folder, so earlier outputs are never overwritten:
`summary.csv`, the figures (see below), `examples.zip` (the `examples/` folder is deleted once zipped; no per-item
raw output is kept) and `config.yaml`, a snapshot of the exact config used (paths made absolute; its first line names
the original file) (run), or the merged `summary.csv`, the figures, `examples.zip`, `config.yaml` and `sources.txt`
(plot; with several `--run`, a later run wins on the same judge × bias, and their `examples.zip` are merged too).
`--out-dir <folder>` (new, empty, or holding only its `config.yaml` snapshot) replaces the timestamped folder;
`plot --allow-missing` skips `--run` folders without a `summary.csv`. `python -m judgebias snapshot --config X`
creates a new run folder holding just that snapshot and prints its path; `job.sh` / `vllm_job.sh` use it at
submission, then run with `--out-dir` that folder, so the config a queued job reads is the one kept with its results.

Each `run` / `plot` (and each `vllm_split.sh` job) gets a 4-character run code, printed as `Run code: k7m2` and
recorded in `<results_dir>/runs.tsv`. `python -m judgebias runs` lists the codes (done / pending);
`python -m judgebias get <code>` zips only that run's result files (summary, figures, `examples.zip`, `config.yaml`,
logs) into `<results_dir>/exports/<code>.zip` for download; `plot --run <code>` also accepts a code.
Nothing is reused between runs: every `run` calls the APIs afresh. Within one run, responses are cached by
(provider, model, effort, params, prompt, repeat) in `results/<YYYYmmdd_HHMMSS>/cache/` so identical calls are made
once; that cache is deleted when the run finishes (or fails).

`examples.zip` holds `examples/<bias>/<judge>/{correct,wrong,invalid}.md`: up to 5 example calls each (input prompt, desired
output, actual output), under a table with that judge's correct/wrong/invalid counts for the bias:

| Bias | Sample | Correct when |
|---|---|---|
| Robustness panels | one perturbed prompt | verdict picks answer 1 (`[[B]]` for Position, where it is shown second); answer 2 for Fallacy-Oversight |
| CoT | one CoT prompt | verdict picks answer 1 |
| Self-Enhancement | judge scoring its own answer | self score within ±0.5 of the other judges' mean |
| Refinement-Aware | score with the "refined" note | score equals the score without the note |

Invalid = a verdict/score needed for the comparison could not be parsed (or the call failed).

### Figures and cost

| File | Shows |
|---|---|
| `bias_grid.png` | the 3×4 grid above: one bar per judge per bias |
| `cost_vs_performance.png` | per bias: mean output tokens per call (x, log) vs. that bias's metric (y) |
| `correctness_grid.png` | same grid, one bar per judge per bias showing Correctness; bars ordered by model (config entry), then effort |
| `cost_vs_correctness.png` | per bias: mean output tokens per call (x, log) vs. Correctness (y) |
| `utility.png` + `utility.csv` | per judge: mean output tokens per call over all its calls (x) vs. mean Correctness across biases (y; error bar = std across biases) |
| `cost_vs_performance_flops.png`, `cost_vs_correctness_flops.png` | the two cost grids with mean FLOPs per call as the cost; only for judges served by `vllm_job.sh` / `vllm_split.sh` |

**Correctness** = `n_correct / samples` over every judged call of that judge × bias (the correct/wrong/invalid rule
above; invalid counts as not correct; `examples/` only shows a few of them), so every bias is on the same 0–1,
higher-is-better scale and can be averaged. **Cost** = output tokens as reported by the API (thinking/reasoning
included), averaged per call. Every call a bias makes counts, cache hits included, so the cost does not depend on what
was cached; Self-Enhancement scoring calls are billed to the judge that scored. In the cost plots, the marker shape shows the effort
and a group's judges are joined by a gray line in effort order (`minimal` < `low` < `medium` < `high` < `xhigh` < `max`).

`summary.csv` carries `group`, `samples`, `n_correct`, `n_wrong` (invalid = the rest), `correctness`, `calls` (calls with a known token count), `out_tokens` (mean output tokens
per call), `flops` and `flops_scale` (below). Token counts are stored in the response cache; calls without a reported count are left out of `out_tokens` (a warning is logged), and a judge × bias with no known count is skipped in
the cost plots.

**FLOPs (vLLM jobs).** `vllm_job.sh` starts vLLM with `--enable-mfu-metrics`, and `config.vllm.yaml` marks its provider
`flops: true`. vLLM only reports a server-wide counter (`vllm:estimated_flops_per_gpu_total` on `/metrics`), and all
judge × bias pairs share the server at once, so each call is estimated with vLLM's own per-step formulas
(`vllm/v1/metrics/perf.py`: QKV/output projections, attention over the context with sliding windows, top-k MoE
experts, logits) from the model's `config.json` and the call's prompt / output tokens (`judgebias/flops.py`). The
estimates are then scaled so that the run's calls add up to the counter's increase over the run (per-GPU value ×
TP GPUs, summed over replicas): `flops_scale` = measured / estimated, and `flops` = mean scaled FLOPs per call, billed
like `out_tokens` (cache hits included). The scale also absorbs what the estimate leaves out, e.g. vLLM's prefix-cache
hits and retried requests. If the counter cannot be read, `flops_scale` is NaN and `flops` is the bare estimate (a
warning is logged); judges of other providers have NaN `flops` and are left out of the FLOP plots.

## Extending

- **New judge:** one line under `judges`. **New endpoint** speaking either protocol: one `providers` entry + one `.env` variable.
- **New bias:** write a `variants(row, T, opts)` function and register a `PerturbationBias` in `judgebias/biases/pairwise.py`
  (or subclass `Bias` for non-pairwise metrics). The runner and plot need no changes.

## Notes

- Anthropic's server-side refusal fallback is deliberately off: it would silently answer with a different model. Refusals count as invalid.
- Cost scales with judges × ~1,500–2,000 calls at `samples_per_bias: 50`.
- Dataset licensing follows the original sources; see `LLM-Judge-Bias-Dataset/README.md`.
