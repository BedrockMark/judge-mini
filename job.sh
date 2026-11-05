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

# Usage:
#   sbatch job.sh                                   # check, then run with config.yaml
#   sbatch job.sh run --config config.yaml --bias position --judge "GPT-oss-20b (low)"
#   sbatch job.sh check --config config.yaml        # any `python -m judgebias` args
# Env must be prepared once on the login node with `uv sync` (compute nodes may lack internet).

set -euo pipefail
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
    python -m judgebias check --config config.yaml
    python -m judgebias run --config config.yaml
fi

echo "end=$(date '+%F %T')"
