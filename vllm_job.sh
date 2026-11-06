#!/bin/bash
#SBATCH --job-name=judgebias_vllm
#SBATCH --output=./job_log/%j.out
#SBATCH --error=./job_log/%j.err
#SBATCH --ntasks=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mark13@ualberta.ca
# GPUs, CPUs, memory and time are set on the sbatch command line by the login-node half below.

# One job runs everything: vLLM server(s) on the job's GPUs + the judgebias client against them.
#
# Run on the LOGIN node with bash (it downloads the model, which needs internet); it then sbatches itself:
#   bash vllm_job.sh                                        # openai/gpt-oss-20b, check + full run
#   bash vllm_job.sh openai/gpt-oss-120b                    # GPU count picked from the model size
#   bash vllm_job.sh openai/gpt-oss-20b --bias position     # extra args go to `judgebias run`
#   GPUS=4 bash vllm_job.sh                                 # more GPUs -> more replicas -> faster
# For a full run prefer vllm_split.sh: same experiment as 3 shorter jobs (start sooner) + an automatic merge.
#
# Inside the job, TP = the smallest tensor-parallel size whose weights fit in 70% of one GPU (the rest is KV
# cache); every further group of TP GPUs runs an independent replica, and the client load-balances over them.
#
# Env overrides (login node): CONFIG (config.vllm.yaml), GPUS (auto), GPU_TYPE (any), GPU_MEM_GB (48, for the
#   auto GPU count), JOB_TIME (24:00:00), CPUS (8/GPU), MEM (32G + 32G/GPU), TP (auto), MAX_MODEL_LEN (32768),
#   MAX_NUM_SEQS (256 per replica), VLLM_EXTRA_ARGS, READY_TIMEOUT (1800 s), VLLM_VENV (.venv-vllm).
# The job runs with a snapshot of CONFIG taken at submission, so CONFIG can be edited and resubmitted for another
# test while this one is still queued. The snapshot is the run's results/<timestamp>/config.yaml (folder created at
# submission, or the --out-dir given) and is kept there with the results.
# Logs: job_log/<jobid>.out (client, last line result=PASS/FAIL), job_log/<jobid>_vllm<i>.log (replica i).

set -uo pipefail

if [[ ${VLLM_JOB_STAGE:-} != run ]]; then
    # ---------------------------------------------------------------- login node: prepare, then submit
    set -e
    cd "$(dirname "$0")"
    MODEL_ID=${1:-${MODEL_ID:-openai/gpt-oss-20b}}
    shift $(( $# > 0 ? 1 : 0 ))
    CONFIG=${CONFIG:-config.vllm.yaml}
    VLLM_VENV=${VLLM_VENV:-.venv-vllm}
    MODEL_DIR=model/${MODEL_ID//\//__}      # openai/gpt-oss-20b -> model/openai__gpt-oss-20b
    export HF_HOME=$PWD/model/.hf_home

    for need in "$VLLM_VENV/bin/vllm" .venv/bin/python; do
        if [[ ! -x $need ]]; then
            echo "$need not found. Prepare once: 'uv sync' (.venv) and vllm_test/README.md (.venv-vllm)." >&2
            exit 1
        fi
    done
    mkdir -p job_log model

    # Fail now, not after the queue, if the config is broken or has no judge for this model
    # (the real URL/key only exist inside the job, hence the placeholders).
    VLLM_BASE_URL=http://placeholder VLLM_API_KEY=placeholder \
        .venv/bin/python -m judgebias check --config "$CONFIG" --model "$MODEL_ID" --no-call --log-dir job_log
    # Freeze the config now (vllm_split.sh passes one it already froze): the job reads it only once it starts.
    # The snapshot is the run's own results/<timestamp>/config.yaml and stays there with the results.
    if ! head -1 "$CONFIG" | grep -q '^# judgebias config snapshot of '; then
        out_dir=()
        for ((i = 1; i <= $#; i++)); do
            case ${!i} in
                --out-dir) j=$((i + 1)); out_dir=(--out-dir "${!j:?--out-dir needs a folder}") ;;
                --out-dir=*) out_dir=(--out-dir "${!i#--out-dir=}") ;;
            esac
        done
        CONFIG=$(.venv/bin/python -m judgebias snapshot --config "$CONFIG" "${out_dir[@]}")
        (( ${#out_dir[@]} )) || set -- "$@" --out-dir "$(dirname "$CONFIG")"
    fi

    # Compute nodes may have no internet. original/ and metal/ are duplicate non-HF-format weights
    # (gpt-oss-120b: +130 GB) that vLLM never reads.
    if [[ ! -f $MODEL_DIR/.download_complete ]]; then
        echo "Downloading $MODEL_ID -> $MODEL_DIR"
        "$VLLM_VENV/bin/python" -c 'import sys; from huggingface_hub import snapshot_download as d
d(sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["original/*", "metal/*"])' "$MODEL_ID" "$MODEL_DIR"
        touch "$MODEL_DIR/.download_complete"
    else
        echo "Using cached model $MODEL_DIR"
    fi
    # gpt-oss tokenizer vocab: openai_harmony would download it at server startup.
    for f in o200k_base.tiktoken cl100k_base.tiktoken; do
        [[ -s model/tiktoken_encodings/$f ]] || curl -sSfL --create-dirs -o "model/tiktoken_encodings/$f" \
            "https://openaipublic.blob.core.windows.net/encodings/$f"
    done

    # Default GPU count: smallest power of two that fits the weights in 70% of each GPU.
    WEIGHTS_GB=$(( $(du -cb "$MODEL_DIR"/*.safetensors | tail -1 | cut -f1) / 1000000000 + 1 ))
    if [[ -z ${GPUS:-} ]]; then
        GPUS=1
        while (( WEIGHTS_GB > GPUS * ${GPU_MEM_GB:-48} * 7 / 10 )); do GPUS=$((GPUS * 2)); done
    fi

    JOB=$(sbatch --parsable --gres="gpu:${GPU_TYPE:+$GPU_TYPE:}$GPUS" --cpus-per-task="${CPUS:-$((8 * GPUS))}" \
        --mem="${MEM:-$((32 + 32 * GPUS))G}" --time="${JOB_TIME:-24:00:00}" \
        --export="ALL,VLLM_JOB_STAGE=run,MODEL_ID=$MODEL_ID,CONFIG=$CONFIG,VLLM_VENV=$VLLM_VENV" \
        vllm_job.sh "$@" | cut -d';' -f1)
    echo "Submitted job $JOB: $MODEL_ID (${WEIGHTS_GB} GB weights) on $GPUS GPU(s), config snapshot $CONFIG"
    cat <<EOF

  squeue -u \$USER                     # status
  tail -f job_log/$JOB.out             # experiment progress
  tail -f job_log/${JOB}_vllm0.log     # vLLM log (replica 0)
  scancel $JOB                         # abort
EOF
    exit 0
fi

# -------------------------------------------------------------------- inside the job
cd "$SLURM_SUBMIT_DIR"
: "${MODEL_ID:?} ${CONFIG:?} ${VLLM_VENV:?}"
MODEL_DIR=model/${MODEL_ID//\//__}
LOG_PREFIX=job_log/${SLURM_JOB_ID}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
PIDS=() URLS=()

finish() {
    trap - EXIT TERM INT
    if (( ${#PIDS[@]} )); then
        kill "${PIDS[@]}" 2>/dev/null
        wait "${PIDS[@]}" 2>/dev/null
    fi
    echo "end=$(date '+%F %T') result=$([[ $1 == 0 ]] && echo PASS || echo FAIL)"
    exit "$1"
}
fail() { echo "FAIL: $*" >&2; finish 1; }
trap 'fail "cancelled or time limit reached"' TERM INT
trap 'finish 1' EXIT   # any unexpected exit still stops the servers

# vLLM JIT-compiles FlashInfer kernels with nvcc + ninja; compute nodes have no /usr/local/cuda.
command -v nvcc >/dev/null || module load cuda
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export HF_HOME=$PWD/model/.hf_home HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TIKTOKEN_ENCODINGS_BASE=$PWD/model/tiktoken_encodings
# Random key: other users on this node could otherwise use the servers via localhost.
export VLLM_API_KEY=$("$VLLM_VENV/bin/python" -c 'import secrets; print(secrets.token_hex(16))')

IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:?no GPU granted}"
NGPU=${#DEVS[@]}
WEIGHTS=$(du -cb "$MODEL_DIR"/*.safetensors | tail -1 | cut -f1)
GPU_MEM=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) * 1024 * 1024 ))
if [[ -z ${TP:-} ]]; then
    TP=1
    while (( TP < NGPU && WEIGHTS / TP > GPU_MEM * 7 / 10 )); do TP=$((TP * 2)); done
fi
(( NGPU % TP == 0 )) || fail "TP=$TP does not divide the $NGPU granted GPUs"
REPLICAS=$((NGPU / TP))
echo "job=$SLURM_JOB_ID node=$(hostname) model=$MODEL_ID gpus=$NGPU tp=$TP replicas=$REPLICAS start=$(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

start_replica() {  # start_replica <i>: vLLM on GPUs [i*TP, (i+1)*TP) of this job
    local i=$1 devs port
    devs=$(IFS=,; echo "${DEVS[*]:i*TP:TP}")
    port=$("$VLLM_VENV/bin/python" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
    echo "replica $i: GPUs $devs port $port log ${LOG_PREFIX}_vllm$i.log"
    # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
    CUDA_VISIBLE_DEVICES=$devs "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
        --served-model-name "$MODEL_ID" \
        --host 127.0.0.1 --port "$port" --api-key "$VLLM_API_KEY" \
        --tensor-parallel-size "$TP" \
        --max-model-len "${MAX_MODEL_LEN:-32768}" \
        --max-num-seqs "${MAX_NUM_SEQS:-256}" \
        ${VLLM_EXTRA_ARGS:-} > "${LOG_PREFIX}_vllm$i.log" 2>&1 &
    PIDS[i]=$! URLS[i]=http://127.0.0.1:$port
}

wait_ready() {  # wait_ready <i>
    local i=$1
    until curl -sf -o /dev/null --max-time 5 "${URLS[i]}/health"; do
        kill -0 "${PIDS[i]}" 2>/dev/null || fail "vLLM replica $i exited during startup; see ${LOG_PREFIX}_vllm$i.log"
        (( SECONDS < READY_TIMEOUT )) || fail "vLLM replica $i not ready after ${READY_TIMEOUT}s"
        sleep 5
    done
    echo "replica $i ready after ${SECONDS}s"
}

# Replica 0 first: it fills the FlashInfer / torch.compile caches, so the others start fast and don't race on them.
start_replica 0
wait_ready 0
for ((i = 1; i < REPLICAS; i++)); do start_replica "$i"; done
for ((i = 1; i < REPLICAS; i++)); do wait_ready "$i"; done

judgebias() {  # python -m judgebias "$@", aborted if any replica dies meanwhile
    python -m judgebias "$@" &
    local pid=$! i
    while kill -0 "$pid" 2>/dev/null; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
                kill "$pid"; wait "$pid"
                fail "vLLM replica $i died during '$1'; see ${LOG_PREFIX}_vllm$i.log"
            fi
        done
        sleep 30 & wait $!   # `wait` (unlike a plain sleep) lets the TERM trap run immediately
    done
    wait "$pid"
}

source .venv/bin/activate
export VLLM_BASE_URL=$(IFS=,; echo "${URLS[*]/%//v1}")   # read by config.vllm.yaml; client balances over all
judgebias check --config "$CONFIG" --model "$MODEL_ID" || finish 1
judgebias run --config "$CONFIG" --model "$MODEL_ID" "$@"
finish $?
