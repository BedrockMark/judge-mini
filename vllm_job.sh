#!/bin/bash
#SBATCH --job-name=jb_vllm             # replaced by <run code>_<model> at submission
#SBATCH --output=./job_log/%j.out
#SBATCH --error=./job_log/%j.err
#SBATCH --ntasks=1
#SBATCH --mail-type=begin,fail        # one mail when a part starts, one more only if it fails / times out
# GPUs, CPUs, memory, time and the mail address (EMAIL in .env) are passed to sbatch by the login-node half below.

# One part of vllm_split.sh: vLLM server(s) on the job's GPUs + the judgebias client against them.
# Full runs go through vllm_split.sh; call this directly only for a small test:
#   bash vllm_job.sh openai/gpt-oss-20b --bias position     # on the LOGIN node (downloads the model); it sbatches itself
# Extra args go to `judgebias run`.
#
# Inside the job, TP = the smallest tensor-parallel size whose weights fit in 70% of one GPU (the rest is KV
# cache); every further group of TP GPUs runs an independent replica, and the client load-balances over them.
#
# Env overrides (login node): CONFIG (config.vllm.yaml), GPUS (auto), GPU_TYPE (any), GPU_MEM_GB (48, for the
#   auto GPU count), JOB_TIME (auto, see below), SEC_PER_SAMPLE (4), CPUS (2 + 2/GPU), MEM (auto, see below),
#   TP (auto), MAX_MODEL_LEN (32768), MAX_NUM_SEQS (256 per replica), VLLM_EXTRA_ARGS, READY_TIMEOUT (1800 s), VLLM_VENV (.venv-vllm).
# Default time = (10 min + SEC_PER_SAMPLE x samples_per_bias x judges of the model / replicas) x 1.25,
# rounded up to 15 min; default RAM = (weights + 8 GB/replica) x 1.25. Measured per vllm_split.sh part (~1/3 run):
#   gpt-oss-20b,  1 GPU, 150 samples x 3 judges: ~30 min wall;  300 samples: 51 min, 0.5 core busy, 18 GB RAM
#   gpt-oss-120b, 2 GPUs (TP=2), 150 x 3: 32 min wall (~4 min startup), 3 cores busy, 70 GB RAM
# i.e. ~3-3.7 s per sample per judge; extra cores only help the startup JIT compile.
# The job runs with a snapshot of CONFIG taken at submission, so CONFIG can be edited and resubmitted for another
# test while this one is still queued. The snapshot is the run's results/<timestamp>/config.yaml (folder created at
# submission, or the --out-dir given) and is kept there with the results.
# The job is named <run code>_<model> (e.g. k7m2_gpt-oss-20b): the code is drawn at submission (or vllm_split.sh's,
# passed as --code) and is the one the results are registered under.
# Logs: job_log/<jobid>.out (client, last line result=PASS/FAIL), job_log/<jobid>_vllm<i>.log (replica i).
# vLLM runs with --enable-mfu-metrics, so the results also get FLOPs per call and the FLOP cost plots.

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
    # Notification address: EMAIL=... in .env, passed to sbatch as --mail-user.
    email=$(sed -n 's/^EMAIL=//p' .env 2>/dev/null | tr -d "\r\"'" | tail -1) || true

    # Fail now, not after the queue, if the config is broken or has no judge for this model
    # (the real URL/key only exist inside the job, hence the placeholders).
    VLLM_BASE_URL=http://placeholder VLLM_API_KEY=placeholder \
        .venv/bin/python -m judgebias check --config "$CONFIG" --model "$MODEL_ID" --no-call --log-dir job_log
    out_dir=() code=
    for ((i = 1; i <= $#; i++)); do
        j=$((i + 1))
        case ${!i} in
            --out-dir) out_dir=(--out-dir "${!j:?--out-dir needs a folder}") ;;
            --out-dir=*) out_dir=(--out-dir "${!i#--out-dir=}") ;;
            --code) code=${!j:?--code needs a run code} ;;
            --code=*) code=${!i#--code=} ;;
        esac
    done
    # The run code (vllm_split.sh passes its own) names the job; the run registers under it when it starts.
    if [[ -z $code ]]; then
        code=$(.venv/bin/python -m judgebias newcode --config "$CONFIG")
        set -- "$@" --code "$code"
    fi
    # Freeze the config now (vllm_split.sh passes one it already froze): the job reads it only once it starts.
    # The snapshot is the run's own results/<timestamp>/config.yaml and stays there with the results.
    if ! head -1 "$CONFIG" | grep -q '^# judgebias config snapshot of '; then
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

    # Replicas the job will run: same TP rule as inside the job, with GPU_MEM_GB as the GPU size.
    tp=${TP:-1}
    if [[ -z ${TP:-} ]]; then
        while (( tp < GPUS && WEIGHTS_GB > tp * ${GPU_MEM_GB:-48} * 7 / 10 )); do tp=$((tp * 2)); done
    fi
    REPLICAS=$(( GPUS / tp > 0 ? GPUS / tp : 1 ))
    # Time: 10 min startup + SEC_PER_SAMPLE per sample per judge of this model per part, shared by the replicas,
    # +25%, rounded up to 15 min, max 24 h.
    read -r SAMPLES NJUDGES <<< "$(.venv/bin/python - "$CONFIG" "$MODEL_ID" <<'EOF'
import sys
from judgebias.config import load_config
cfg = load_config(sys.argv[1])
print(cfg.samples_per_bias, max(1, sum(j.model == sys.argv[2] for j in cfg.judges)))
EOF
)"
    MINUTES=$(( (600 + ${SEC_PER_SAMPLE:-4} * SAMPLES * NJUDGES / REPLICAS) * 5 / 4 / 60 ))
    MINUTES=$(( (MINUTES + 14) / 15 * 15 ))
    (( MINUTES <= 1440 )) || MINUTES=1440
    # RAM: the weights (read through the page cache, which counts against --mem) + 8 GB per replica, +25%.
    MEM_GB=$(( ((WEIGHTS_GB + 8 * REPLICAS) * 5 / 4 + 3) / 4 * 4 ))
    JOB=$(sbatch --parsable ${email:+"--mail-user=$email"} --job-name="${code}_${MODEL_ID##*/}" --gres="gpu:${GPU_TYPE:+$GPU_TYPE:}$GPUS" --cpus-per-task="${CPUS:-$((2 + 2 * GPUS))}" \
        --mem="${MEM:-${MEM_GB}G}" --time="${JOB_TIME:-$((MINUTES / 60)):$(printf %02d $((MINUTES % 60))):00}" \
        --export="ALL,VLLM_JOB_STAGE=run,MODEL_ID=$MODEL_ID,CONFIG=$CONFIG,VLLM_VENV=$VLLM_VENV" \
        vllm_job.sh "$@" | cut -d';' -f1)
    echo "Submitted job $JOB ($code): $MODEL_ID (${WEIGHTS_GB} GB weights) on $GPUS GPU(s), config snapshot $CONFIG"
    echo "  time ${JOB_TIME:-$((MINUTES / 60)):$(printf %02d $((MINUTES % 60))):00}, ${CPUS:-$((2 + 2 * GPUS))} CPUs, ${MEM:-${MEM_GB}G} RAM ($SAMPLES samples x $NJUDGES judges, $REPLICAS replica(s))"
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
        --enable-mfu-metrics \
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
# A `flops: true` provider estimates each call's FLOPs from this config.json and scales them to the servers'
# --enable-mfu-metrics counter (per GPU, x TP GPUs per replica).
export VLLM_MODEL_DIR=$PWD/$MODEL_DIR VLLM_TP=$TP
judgebias check --config "$CONFIG" --model "$MODEL_ID" || finish 1
judgebias run --config "$CONFIG" --model "$MODEL_ID" "$@"
finish $?
