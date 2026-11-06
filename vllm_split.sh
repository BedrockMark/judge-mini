#!/bin/bash
# One model's full run as a few shorter vllm_job.sh jobs (one per bias group) + a small CPU job that merges them.
# Short jobs are scheduled sooner (backfill), parts run side by side if GPUs are free, and a time-out loses one part.
#
# Run on the LOGIN node with bash, like vllm_job.sh:
#   bash vllm_split.sh                                   # openai/gpt-oss-20b
#   bash vllm_split.sh openai/gpt-oss-120b
#   JOB_TIME=06:00:00 GPUS=2 bash vllm_split.sh          # vllm_job.sh env vars apply to every part
#   bash vllm_split.sh openai/gpt-oss-20b --judge "GPT-oss-20b (low)"   # extra args go to every `judgebias run`
#   bash vllm_split.sh --resume k7m2                     # rerun only the parts of run k7m2 that timed out / failed /
#   JOB_TIME=12:00:00 bash vllm_split.sh --resume k7m2   #   were cancelled, then merge again (same model, args, env
#                                                        #   as before unless overridden like here)
#
# The groups keep biases that send identical prompts together (within one run those are asked once):
# the 5 alignment-set biases share their 4 unperturbed repeats, authority + sentiment and verbosity +
# fallacy_oversight share theirs. Splitting them apart would cost ~60% more calls; these 3 groups cost ~7% more
# than one job and need about the same output tokens each. Every part runs all of the model's judges, which
# self_enhancement needs (judges score each other).
#
# Env overrides: PARTS (groups separated by ';', biases by ','), JOB_TIME (08:00:00 per part), CONFIG
#   (config.vllm.yaml), plus everything vllm_job.sh reads.
# Output: <results_dir>/split_<timestamp>_<model>/part<i>/ per job, merged/ (full result, like a single run),
#   config.yaml (snapshot of CONFIG at submission: every part, the merge and --resume use it, so CONFIG can be
#   edited and another split submitted right away) and split.vars (what --resume needs). The split's run code is printed at the end and listed by
#   `python -m judgebias runs`; once merged, `python -m judgebias get <code>` zips merged/'s result files.
# The merge runs even if a part failed (without it); `--resume <code>` then reruns just that part and re-merges.

set -euo pipefail
cd "$(dirname "$0")"
# Notification address: EMAIL=... in .env (sbatch reads SBATCH_MAIL_USER as --mail-user).
email=$(sed -n 's/^EMAIL=//p' .env 2>/dev/null | tr -d '\r"' | tail -1) || true
[[ -z $email || -n ${SBATCH_MAIL_USER:-} ]] || export SBATCH_MAIL_USER=$email
# vllm_job.sh settings a resume reuses unless they are set again.
SAVED_ENV=(JOB_TIME GPUS GPU_TYPE GPU_MEM_GB CPUS MEM TP MAX_MODEL_LEN MAX_NUM_SEQS VLLM_EXTRA_ARGS READY_TIMEOUT VLLM_VENV)
GIVEN_JOB_TIME=${JOB_TIME:-}
JOBS=() MERGE_JOB=

if [[ ${1:-} == --resume ]]; then
    CODE=${2:?usage: bash vllm_split.sh --resume <run code>}
    OUT=$(.venv/bin/python - "${CONFIG:-config.vllm.yaml}" "$CODE" <<'EOF'
import sys
from judgebias import registry
from judgebias.config import load_config
results_dir = load_config(sys.argv[1]).results_dir
entry = registry.lookup(results_dir, sys.argv[2])
if entry is None or entry["kind"] != "split":
    sys.exit(f"No vllm_split.sh run with code '{sys.argv[2]}' in {registry.registry_path(results_dir)}")
print(registry.result_dir(results_dir, entry).parent)
EOF
)
    [[ -f $OUT/split.vars ]] || { echo "$OUT/split.vars not found; cannot resume." >&2; exit 1; }
    source "$OUT/split.vars"   # MODEL_ID CONFIG PARTS EXTRA JOBS MERGE_JOB CODE, + earlier env unless set now
else
    MODEL_ID=${1:-openai/gpt-oss-20b}
    shift $(( $# > 0 ? 1 : 0 ))
    CONFIG=${CONFIG:-config.vllm.yaml}
    PARTS=${PARTS:-position,compassion_fade,bandwagon,distraction,diversity;authority,sentiment;verbosity,fallacy_oversight,cot,refinement_aware,self_enhancement}
    EXTRA=("$@")
fi
export CONFIG JOB_TIME=${JOB_TIME:-08:00:00}
IFS=';' read -ra GROUPS_ <<< "$PARTS"

# Fail before submitting anything if a bias is misspelled or one the config selects is in no group.
RESULTS_DIR=$(.venv/bin/python - "$CONFIG" "$PARTS" <<'EOF'
import sys
from judgebias.config import load_config
cfg = load_config(sys.argv[1])
listed = [b for g in sys.argv[2].split(";") for b in g.split(",") if b]
unknown = sorted(set(listed) - set(cfg.selected_biases()))
missing = [b for b in cfg.selected_biases() if b not in listed]
if unknown or missing:
    sys.exit(f"PARTS: not selected by {sys.argv[1]} or unknown: {unknown}; in no group: {missing}")
print(cfg.results_dir)
EOF
)
if [[ -z ${CODE:-} ]]; then
    OUT=$RESULTS_DIR/split_$(date +%Y%m%d_%H%M%S)_${MODEL_ID//\//__}
    CODE=$(.venv/bin/python -c 'import sys; from pathlib import Path; from judgebias import registry
print(registry.register(Path(sys.argv[1]), Path(sys.argv[2]), "split", sys.argv[3]))' \
        "$RESULTS_DIR" "$OUT" "config=$(basename "$CONFIG") model=$MODEL_ID parts=$PARTS${EXTRA[*]:+ args=${EXTRA[*]}}")
    CONFIG=$(.venv/bin/python -m judgebias snapshot --config "$CONFIG" "$OUT/config.yaml")
fi

save() {  # everything --resume needs; rewritten after every submission so job IDs are never lost
    {
        declare -p MODEL_ID CONFIG PARTS CODE EXTRA JOBS MERGE_JOB   # sourced at top level, so these are globals
        for v in "${SAVED_ENV[@]}"; do
            if [[ -n ${!v:-} ]]; then printf '[[ -n ${%s:-} ]] || export %s=%q\n' "$v" "$v" "${!v}"; fi
        done
    } > "$OUT/split.vars"
}

DEPS=() RUNS=() TIMED_OUT=()
for i in "${!GROUPS_[@]}"; do
    n=$((i + 1)) part=$OUT/part$((i + 1))
    RUNS+=(--run "$part")
    IFS=',' read -ra biases <<< "${GROUPS_[i]}"
    if [[ -n ${JOBS[i]:-} ]]; then  # resuming: keep finished and still-queued parts
        if [[ -f $part/summary.csv ]]; then
            echo "== part$n: done (job ${JOBS[i]})"
            continue
        fi
        state=$(squeue -h -j "${JOBS[i]}" -o %T 2>/dev/null || true)
        if [[ -n $state ]]; then
            echo "== part$n: still ${state,,} (job ${JOBS[i]}), not resubmitted"
            DEPS+=("${JOBS[i]}")
            continue
        fi
        state=$(sacct -X -n -P -j "${JOBS[i]}" -o State 2>/dev/null | head -1 || true)
        echo "== part$n: job ${JOBS[i]} ended ${state:-without a result} -> resubmitting"
        [[ $state == TIMEOUT* ]] && TIMED_OUT+=("part$n")
        rm -rf "$part"
    fi
    args=()
    for b in "${biases[@]}"; do args+=(--bias "$b"); done
    echo "== part$n: ${biases[*]}"
    submitted=$(bash vllm_job.sh "$MODEL_ID" "${args[@]}" --out-dir "$part" --part-of "$CODE" "${EXTRA[@]}" | tee /dev/stderr)
    job=$(sed -n 's/^Submitted job \([0-9]*\).*/\1/p' <<< "$submitted")
    [[ -n $job ]] || { echo "Could not read the job ID of part$n; not submitting the merge." >&2; save; exit 1; }
    JOBS[i]=$job DEPS+=("$job")
    save
done

if [[ -n $MERGE_JOB ]]; then  # resuming: the earlier merge is stale (or would start before the reruns end)
    if (( ${#DEPS[@]} == 0 )) && [[ -f $OUT/merged/sources.txt ]] \
        && (( $(wc -l < "$OUT/merged/sources.txt") == ${#GROUPS_[@]} )); then
        echo "All parts are done and merged; nothing to rerun. Results: $OUT/merged/"
        exit 0
    fi
    scancel "$MERGE_JOB" 2>/dev/null || true
    rm -rf "$OUT/merged"
fi
# afterany: merge whatever parts finished, even if one failed or timed out.
MERGE_JOB=$(sbatch --parsable --job-name=judgebias_merge ${DEPS[@]:+--dependency="afterany:$(IFS=:; echo "${DEPS[*]}")"} \
    --time=00:30:00 --cpus-per-task=1 --mem=4G \
    job.sh plot --config "$CONFIG" --allow-missing "${RUNS[@]}" --out-dir "$OUT/merged" --part-of "$CODE" | cut -d';' -f1)
save

if (( ${#TIMED_OUT[@]} )) && [[ -z $GIVEN_JOB_TIME ]]; then
    echo "WARNING: ${TIMED_OUT[*]} timed out and is rerun with the same JOB_TIME=$JOB_TIME; to give it more," \
         "scancel it and run: JOB_TIME=<longer> bash vllm_split.sh --resume $CODE" >&2
fi
cat <<EOF

Run code: $CODE
Parts: ${JOBS[*]} (run side by side as GPUs free up)   merge: $MERGE_JOB (starts when all parts end)
Results: $OUT/merged/   (job_log/$MERGE_JOB.err warns about any missing part)

  python -m judgebias runs --config $CONFIG          # done / pending
  python -m judgebias get $CODE --config $CONFIG     # -> $RESULTS_DIR/exports/$CODE.zip
  bash vllm_split.sh --resume $CODE                   # a part timed out / failed / was cancelled: rerun only it
EOF
