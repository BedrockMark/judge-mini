#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --output=./job_log/%j.out
#SBATCH --error=./job_log/%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00                 # short + small jobs queue fastest (backfill)

# vLLM OpenAI-compatible server. Submitted by vllm_test.sh, which sets via --export:
#   MODEL_DIR, SERVED_NAME, VLLM_API_KEY, VLLM_VENV
# Optional: MAX_MODEL_LEN (default 4096), VLLM_EXTRA_ARGS (appended to `vllm serve`).
# Writes job_log/vllm_<jobid>.endpoint (the base URL) for the check job to find the server.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MODEL_DIR:?} ${SERVED_NAME:?} ${VLLM_API_KEY:?} ${VLLM_VENV:?}"

source "$VLLM_VENV/bin/activate"
export HF_HOME=$PWD/model/.hf_home HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
ENDPOINT_FILE=job_log/vllm_${SLURM_JOB_ID}.endpoint
trap 'rm -f "$ENDPOINT_FILE"' EXIT
echo "http://$(hostname):$PORT" > "$ENDPOINT_FILE.tmp" && mv "$ENDPOINT_FILE.tmp" "$ENDPOINT_FILE"

NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")   # GPUs or MIG slices granted by SLURM
echo "job=$SLURM_JOB_ID node=$(hostname) port=$PORT gpus=$NGPU model=$MODEL_DIR start=$(date '+%F %T')"
nvidia-smi

# Not exec'd, so the EXIT trap still removes the endpoint file.
# shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --api-key "$VLLM_API_KEY" \
    --tensor-parallel-size "$NGPU" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    ${VLLM_EXTRA_ARGS:-}
