#!/bin/bash
#SBATCH --job-name=judgebias
#SBATCH --output=./job_log/%j.out       # stdout (job_log/ must exist before sbatch)
#SBATCH --error=./job_log/%j.err        # stderr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # API calls only, no GPU needed
#SBATCH --mem=8G
#SBATCH --time=48:00:00                 # 3 judges x all biases took ~1.5h locally
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mark13@ualberta.ca

# Usage (login node; `bash` freezes the config now and then submits itself, so the config can be edited and another
# test submitted while this one is still queued):
#   bash job.sh                                     # check, then run with config.yaml
#   bash job.sh run --config config.yaml --bias position --judge "GPT-oss-20b (low)"
#   bash job.sh check --config config.yaml          # any `python -m judgebias` args
#   SBATCH_ARGS="--time=08:00:00" bash job.sh       # extra sbatch options
# For run / plot (and the default check + run) the snapshot is the run's own results/<timestamp>/config.yaml (folder
# created now, filled when the job runs; --out-dir picks it instead) and is kept there with the results; other
# commands snapshot to job_log/config/.
# `sbatch job.sh ...` also works, but then the job reads the config file only when it starts.
# Env must be prepared once on the login node with `uv sync` (compute nodes may lack internet).

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    # ------------------------------------------------------------ login node: snapshot the config, then submit
    cd "$(dirname "$0")"
    mkdir -p job_log
    args=("$@") config=config.yaml at= out_dir=() out=
    for i in "${!args[@]}"; do
        case ${args[i]} in
            --config) at=$((i + 1)); config=${args[at]:?--config needs a file} ;;
            --config=*) at=$i; config=${args[i]#--config=} ;;
            --out-dir) out_dir=(--out-dir "${args[i + 1]:?--out-dir needs a folder}") ;;
            --out-dir=*) out_dir=(--out-dir "${args[i]#--out-dir=}") ;;
        esac
    done
    if (( ${#args[@]} == 0 )) || [[ ${args[0]} == run || ${args[0]} == plot ]]; then
        snap=$(.venv/bin/python -m judgebias snapshot --config "$config" "${out_dir[@]}")  # <out folder>/config.yaml
        out=$(dirname "$snap")
    else
        snap=$(.venv/bin/python -m judgebias snapshot --config "$config" \
            "job_log/config/$(date +%Y%m%d_%H%M%S)_$$_$(basename "$config")")
    fi
    if (( ${#args[@]} == 0 )); then
        export CONFIG=$snap OUT_DIR=$out          # default check + run below
    else
        if [[ -n $at ]]; then
            args[at]=$([[ ${args[at]} == --config=* ]] && echo "--config=$snap" || echo "$snap")
        else
            args+=(--config "$snap")
        fi
        if [[ -n $out ]] && (( ${#out_dir[@]} == 0 )); then args+=(--out-dir "$out"); fi
    fi
    # shellcheck disable=SC2086  # SBATCH_ARGS is intentionally word-split
    sbatch ${SBATCH_ARGS:-} job.sh "${args[@]}"
    echo "config snapshot: $snap"
    exit 0
fi

# -------------------------------------------------------------------- inside the job
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found in $(pwd); run 'uv sync' on the login node first." >&2
    exit 1
fi
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

echo "job=${SLURM_JOB_ID:-local} node=$(hostname) dir=$(pwd) start=$(date '+%F %T')"

if [[ $# -gt 0 ]]; then
    python -m judgebias "$@"
else
    # Fail fast (bad key / model ID / endpoint unreachable) before the long run.
    python -m judgebias check --config "${CONFIG:-config.yaml}"
    python -m judgebias run --config "${CONFIG:-config.yaml}" ${OUT_DIR:+--out-dir "$OUT_DIR"}
fi

echo "end=$(date '+%F %T')"
