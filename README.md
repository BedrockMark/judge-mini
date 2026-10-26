# reasoning-judge-bias-mini

A small, config-driven reproduction of the bias benchmark from
[*Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge*](https://arxiv.org/abs/2410.02736) (CALM).
You list the judges you want — each a **(provider, model, thinking effort)** triple — and the tool runs the
paper's 12 bias tests on only those judges and draws a 3×4 grid like `Goal.png`.

It reuses the paper's datasets and prompt builders from `LLM-Judge-Bias-Dataset/` (not copied).

## Setup

```bash
uv sync                      # Python >= 3.12
cp .env.example .env         # then fill in the keys you need
```

`.env` holds secrets; `config.yaml` only holds the **name** of the variable each provider uses
(`api_key_env`), so keys never appear in the config.

## Usage

```bash
uv run python -m judgebias check --config config.yaml   # validate config + keys, then one test call per judge
uv run python -m judgebias run   --config config.yaml   # run judges x biases -> results/summary.csv + bias_grid.png
uv run python -m judgebias plot  --config config.yaml   # redraw the figure from results/summary.csv
uv run pytest                                           # tests (no API calls)
```

- `run` accepts `--judge "<label>"` and `--bias <name>` (both repeatable) to run a subset. Partial runs are merged into `summary.csv`.
- `check --no-call` skips the live test calls.
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
| `concurrency` | Max simultaneous API calls |
| `cr_repeats` | Times the unperturbed prompt is judged, for Consistency Rate (≥ 2) |
| `max_tokens` | Per-response cap (thinking + answer) |
| `providers` | `name: {protocol: openai\|anthropic, base_url, api_key_env}`; any OpenAI-compatible endpoint works (use `token_param: max_tokens` if it needs it) |
| `judges` | `{label, provider, model, effort, params?, color?}` — only these are ever called |
| `biases` | `all` or a list of bias names |
| `bias_options` | Per-bias knobs (bandwagon `number`, diversity `groups`, compassion-fade `model_names`, …) |

`effort` maps to `reasoning_effort` (OpenAI protocol) or `output_config.effort` (Anthropic protocol);
`null` sends nothing. `params` is passed through verbatim (e.g. `thinking: {type: between_tools}` to turn
Sonnet 5.5 thinking off). Judge labels must be unique.

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
unparseable verdicts count as non-robust and are tallied in `n_invalid`. The red dashed line is the mean
Consistency Rate across judges for RR panels, and the mean metric across judges for the others.
Perturbations that target one answer always target the rejected one (answer 2), except Fallacy-Oversight and
Sentiment (answer 1). Dataset rows missing a perturbed field are skipped, not invented.

## Layout

```
config.yaml / config.fake.yaml   real and offline configs
judgebias/
  config.py       config models, .env loading, key matching
  check.py        pre-flight check with live test calls
  llm/            protocol adapters (openai, anthropic, fake), JSONL response cache
  context.py      cached, concurrency-limited `ask` shared by all biases
  data.py, prompts.py, parse.py, metrics.py
  biases/         one module/registration per panel; registry is `BIASES`
  runner.py       judge x bias loop -> results/
  plot.py         3x4 figure
tests/            pytest suite (no network)
```

Outputs go to `results/`: `summary.csv`, `bias_grid.png`, per-item `raw/<judge>/<bias>.jsonl`, and
`cache/responses.jsonl`. Responses are cached by (provider, model, effort, params, prompt, repeat), so
re-running costs nothing for finished work and failed calls are retried.

## Extending

- **New judge:** one line under `judges`. **New endpoint** speaking either protocol: one `providers` entry + one `.env` variable.
- **New bias:** write a `variants(row, T, opts)` function and register a `PerturbationBias` in `judgebias/biases/pairwise.py`
  (or subclass `Bias` for non-pairwise metrics). The runner and plot need no changes.

## Notes

- Anthropic's server-side refusal fallback is deliberately off: it would silently answer with a different model. Refusals count as invalid.
- Cost scales with judges × ~1,500–2,000 calls at `samples_per_bias: 50`.
- Dataset licensing follows the original sources; see `LLM-Judge-Bias-Dataset/README.md`.
