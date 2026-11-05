#!/bin/bash
#SBATCH --job-name=vllm_check
#SBATCH --output=./job_log/%j.out
#SBATCH --error=./job_log/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00                 # covers queue overlap + READY_TIMEOUT

# Connectivity/response check for the vLLM server job. Submitted by vllm_test.sh with
# --dependency=after:$SERVE_JOB, so it starts once the server job is running, then:
#   1. waits for job_log/vllm_<SERVE_JOB>.endpoint (written by the server job),
#   2. polls <url>/health until vLLM has loaded the model (READY_TIMEOUT s, default 600),
#   3. runs vllm_test/vllm_client_test.py against <url>/v1,
#   4. scancels the server unless KEEP_SERVER=1.
# Set via --export: SERVE_JOB, SERVED_NAME, VLLM_API_KEY, VLLM_VENV, KEEP_SERVER.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${SERVE_JOB:?} ${SERVED_NAME:?} ${VLLM_API_KEY:?} ${VLLM_VENV:?}"

ENDPOINT_FILE=job_log/vllm_${SERVE_JOB}.endpoint
DEADLINE=$((SECONDS + ${READY_TIMEOUT:-600}))
echo "job=$SLURM_JOB_ID node=$(hostname) server_job=$SERVE_JOB start=$(date '+%F %T')"

server_alive() { [[ $(squeue -h -j "$SERVE_JOB" -o %T 2>/dev/null) == RUNNING ]]; }

finish() {
    if [[ ${KEEP_SERVER:-0} != 1 ]] && server_alive; then
        echo "Stopping server job $SERVE_JOB"
        scancel "$SERVE_JOB"
    fi
    echo "end=$(date '+%F %T') result=$([[ $1 == 0 ]] && echo PASS || echo FAIL)"
    exit "$1"
}

wait_for() {  # wait_for <description> <command...>: retry every 5 s until success, server death, or deadline
    local what=$1; shift
    echo "Waiting for $what ..."
    until "$@"; do
        if ! server_alive; then
            echo "FAIL: server job $SERVE_JOB is no longer running; see job_log/$SERVE_JOB.err" >&2
            finish 1
        fi
        if (( SECONDS > DEADLINE )); then
            echo "FAIL: timed out after ${READY_TIMEOUT:-600}s waiting for $what" >&2
            finish 1
        fi
        sleep 5
    done
    echo "  ready after ${SECONDS}s"
}

wait_for "endpoint file $ENDPOINT_FILE" test -s "$ENDPOINT_FILE"
URL=$(<"$ENDPOINT_FILE")
echo "Server URL: $URL"

wait_for "$URL/health" curl -sf -o /dev/null --max-time 5 "$URL/health"

"$VLLM_VENV/bin/python" vllm_test/vllm_client_test.py --base-url "$URL/v1" --model "$SERVED_NAME"
finish $?
