#!/bin/bash
# Smoke-test vLLM on the cluster: submit the GPU server job, then a CPU check job that
# starts once the server job has started (--dependency=after) and probes the endpoint.
#
# Run on the LOGIN node (not with sbatch) -- it downloads the model, which needs internet:
#   bash vllm_test/vllm_test.sh                                   # default model
#   bash vllm_test/vllm_test.sh Qwen/Qwen2.5-1.5B-Instruct        # any HF model ID
#   KEEP_SERVER=1 bash vllm_test/vllm_test.sh                     # leave the server running after the check
#
# Results: job_log/<check_jobid>.out (check) and job_log/<serve_jobid>.out/.err (server).

set -euo pipefail
cd "$(dirname "$0")/.."    # project root: model/, job_log/ and .venv-vllm live there; any cwd works

MODEL_ID=${1:-${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}}
VLLM_VENV=${VLLM_VENV:-.venv-vllm}
MODEL_DIR=model/${MODEL_ID//\//__}          # Qwen/Qwen2.5-0.5B-Instruct -> model/Qwen__Qwen2.5-0.5B-Instruct
export HF_HOME=$PWD/model/.hf_home          # keep every HF cache/download inside ./model

if [[ ! -x $VLLM_VENV/bin/vllm ]]; then
    echo "vLLM not found in $VLLM_VENV. Create it once (see vllm_test/README.md):" >&2
    echo "  uv venv $VLLM_VENV --python 3.12 && uv pip install --python $VLLM_VENV vllm" >&2
    exit 1
fi
PY=$VLLM_VENV/bin/python
mkdir -p job_log model

# Download on the login node; compute nodes may have no internet. Marker = finished download.
if [[ ! -f $MODEL_DIR/.download_complete ]]; then
    echo "Downloading $MODEL_ID -> $MODEL_DIR"
    "$PY" -c 'import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2])' \
        "$MODEL_ID" "$MODEL_DIR"
    touch "$MODEL_DIR/.download_complete"
else
    echo "Using cached model $MODEL_DIR"
fi

# Random per-run API key so other cluster users can't use the server.
VLLM_API_KEY=${VLLM_API_KEY:-$("$PY" -c 'import secrets; print(secrets.token_hex(16))')}
COMMON=ALL,VLLM_VENV=$VLLM_VENV,VLLM_API_KEY=$VLLM_API_KEY,SERVED_NAME=$MODEL_ID

# GPU_GRES overrides the GPU request, e.g. a MIG slice if the cluster offers one (see README).
SERVE_JOB=$(sbatch --parsable ${GPU_GRES:+--gres="$GPU_GRES"} --export="$COMMON,MODEL_DIR=$MODEL_DIR" \
    vllm_test/vllm_serve.sh | cut -d';' -f1)
echo "Submitted server job $SERVE_JOB"

# after:<id> = begin once job <id> has STARTED (not finished).
CHECK_JOB=$(sbatch --parsable --dependency=after:"$SERVE_JOB" --kill-on-invalid-dep=yes \
    --export="$COMMON,SERVE_JOB=$SERVE_JOB,KEEP_SERVER=${KEEP_SERVER:-0}" vllm_test/vllm_check.sh | cut -d';' -f1)
echo "Submitted check job  $CHECK_JOB (starts after $SERVE_JOB starts)"

cat <<EOF

  squeue -u \$USER                         # status
  tail -f job_log/$CHECK_JOB.out           # check result
  tail -f job_log/$SERVE_JOB.err           # vLLM server log
  scancel $SERVE_JOB $CHECK_JOB            # abort
EOF
