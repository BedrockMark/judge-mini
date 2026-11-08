#!/bin/bash
# Quick check that a model runs on vLLM here: download it into ./model, then a 15-min GPU job (test_job.sh) starts
# vLLM with the same GPUs / CPUs / RAM and server flags as a vllm_split.sh part and sends one request.
# Mail only when the job ends or fails. Result: last line of job_log/<jobid>.out (PASS / FAIL).
#
# Run on the LOGIN node with bash:
#   bash test_model.sh Qwen/Qwen3-8B
# vllm_job.sh's env overrides apply (GPUS, GPU_TYPE, GPU_MEM_GB, CPUS, MEM, TP, MAX_MODEL_LEN, ...), except JOB_TIME.

set -euo pipefail
cd "$(dirname "$0")"
MODEL_ID=${1:?usage: bash test_model.sh <huggingface model id>}
VLLM_VENV=${VLLM_VENV:-.venv-vllm}
MODEL_DIR=model/${MODEL_ID//\//__}
export HF_HOME=$PWD/model/.hf_home
mkdir -p job_log model
email=$(sed -n 's/^EMAIL=//p' .env 2>/dev/null | tr -d "\r\"'" | tail -1) || true

# Same download as vllm_job.sh.
if [[ ! -f $MODEL_DIR/.download_complete ]]; then
    "$VLLM_VENV/bin/python" -c 'import sys; from huggingface_hub import snapshot_download as d
d(sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["original/*", "metal/*"])' "$MODEL_ID" "$MODEL_DIR"
    touch "$MODEL_DIR/.download_complete"
fi
for f in o200k_base.tiktoken cl100k_base.tiktoken; do
    [[ -s model/tiktoken_encodings/$f ]] || curl -sSfL --create-dirs -o "model/tiktoken_encodings/$f" \
        "https://openaipublic.blob.core.windows.net/encodings/$f"
done

# Same GPU / RAM / CPU rules as vllm_job.sh.
WEIGHTS_GB=$(( $(du -cb "$MODEL_DIR"/*.safetensors | tail -1 | cut -f1) / 1000000000 + 1 ))
if [[ -z ${GPUS:-} ]]; then
    GPUS=1
    while (( WEIGHTS_GB > GPUS * ${GPU_MEM_GB:-48} * 7 / 10 )); do GPUS=$((GPUS * 2)); done
fi
tp=${TP:-1}
if [[ -z ${TP:-} ]]; then
    while (( tp < GPUS && WEIGHTS_GB > tp * ${GPU_MEM_GB:-48} * 7 / 10 )); do tp=$((tp * 2)); done
fi
REPLICAS=$(( GPUS / tp > 0 ? GPUS / tp : 1 ))
MEM_GB=$(( ((WEIGHTS_GB + 8 * REPLICAS) * 5 / 4 + 3) / 4 * 4 ))

sbatch ${email:+"--mail-user=$email"} --job-name="test_${MODEL_ID##*/}" --gres="gpu:${GPU_TYPE:+$GPU_TYPE:}$GPUS" \
    --cpus-per-task="${CPUS:-$((2 + 2 * GPUS))}" --mem="${MEM:-${MEM_GB}G}" \
    --export="ALL,MODEL_ID=$MODEL_ID,VLLM_VENV=$VLLM_VENV" test_job.sh
