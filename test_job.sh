#!/bin/bash
#SBATCH --job-name=jb_test             # replaced by test_<model> at submission
#SBATCH --output=./job_log/%j.out      # vLLM log + result, last line PASS / FAIL
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --mail-type=end,fail
# Submitted by test_model.sh (GPUs, CPUs, memory, mail address): vLLM as in vllm_job.sh, then one request.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${MODEL_ID:?} ${VLLM_VENV:?}"
MODEL_DIR=model/${MODEL_ID//\//__}

command -v nvcc >/dev/null || module load cuda
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export HF_HOME=$PWD/model/.hf_home HF_HUB_OFFLINE=1 TIKTOKEN_ENCODINGS_BASE=$PWD/model/tiktoken_encodings
KEY=$("$VLLM_VENV/bin/python" -c 'import secrets; print(secrets.token_hex(16))')
PORT=$("$VLLM_VENV/bin/python" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')

# Same TP rule as vllm_job.sh; one replica is enough for a test.
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:?no GPU granted}"
WEIGHTS=$(du -cb "$MODEL_DIR"/*.safetensors | tail -1 | cut -f1)
GPU_MEM=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) * 1024 * 1024 ))
if [[ -z ${TP:-} ]]; then
    TP=1
    while (( TP < ${#DEVS[@]} && WEIGHTS / TP > GPU_MEM * 7 / 10 )); do TP=$((TP * 2)); done
fi

# shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${DEVS[*]:0:TP}") "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$PORT" --api-key "$KEY" \
    --tensor-parallel-size "$TP" --max-model-len "${MAX_MODEL_LEN:-32768}" --max-num-seqs "${MAX_NUM_SEQS:-256}" \
    --enable-mfu-metrics ${VLLM_EXTRA_ARGS:-} 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

until curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/health"; do
    kill -0 $PID 2>/dev/null || { echo FAIL: vLLM exited during startup; exit 1; }
    sleep 5
done
curl -sf --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 256}" \
    || { echo; echo FAIL: request failed; exit 1; }
echo; echo "PASS after ${SECONDS}s"
