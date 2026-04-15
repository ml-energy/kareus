#!/usr/bin/env bash
# Toy Bayesian optimization test on 4 GPUs (TP=2, CP=1).
# Matches the config from tests/toy/run_megatron.sh:
#   Llama 3.2 3B, TP=2, CP=1, MBS=4, SEQ=2048.
# Splits 6 partitions across two GPU pairs run in parallel.
#
# Usage:
#   bash test_bayesian.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAYESIAN_DIR="${SCRIPT_DIR}/../bayesian"
cd "$BAYESIAN_DIR"

MODEL=llama3.2_3b
TP=2
CP=1
BS=4
SEQ=2048
GPU_TYPE=A40
N_INIT=16
BATCHES=2
ACQ_BATCH=16

PARTS_A=(fwd_attn fwd_mlp)
PARTS_B=(bwd_attn bwd_mlp)

TAG="${MODEL}_cp${CP}_tp${TP}_bs${BS}_seq${SEQ}"
LOGS_DIR="logs/${MODEL}/cp${CP}-tp${TP}-bs${BS}-seq${SEQ}"

wait_all() {
    local failed=0
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        echo "ERROR: one or more parallel jobs failed" >&2
        exit 1
    fi
}

run_parts() {
    local gpus=$1; shift
    local parts=("$@")
    for part in "${parts[@]}"; do
        local log_dir="${LOGS_DIR}/${part}"
        mkdir -p "$log_dir"
        echo ">>> ${part} (GPUs ${gpus})"
        CUDA_VISIBLE_DEVICES=$gpus python "partitions/${part}/bo_search.py" \
            -m "$MODEL" -w "$TP" -tp "$TP" -cp "$CP" -b "$BS" -s "$SEQ" --gpu_type "$GPU_TYPE" \
            --n_init "$N_INIT" --batches "$BATCHES" --acq_batch "$ACQ_BATCH" \
            > "${log_dir}/bo_${TAG}.log" 2>&1
    done
}

echo "===== Toy Bayesian Test (${TAG}) ====="
echo "  Group A (GPUs 0,1): ${PARTS_A[*]}"
echo "  Group B (GPUs 2,3): ${PARTS_B[*]}"

pids=()
run_parts 0,1 "${PARTS_A[@]}" & pids+=($!)
run_parts 2,3 "${PARTS_B[@]}" & pids+=($!)
wait_all "${pids[@]}"

echo "===== Nonpartition (GPUs 0,1) ====="
NP_LOG_DIR="${LOGS_DIR}/nonpartition"
mkdir -p "$NP_LOG_DIR"
CUDA_VISIBLE_DEVICES=0,1 python "nonpartition/profile_nonpartition.py" \
    -m "$MODEL" -w "$TP" -tp "$TP" -cp "$CP" -b "$BS" -s "$SEQ" --gpu_type "$GPU_TYPE" \
    > "${NP_LOG_DIR}/nonpartition_${TAG}.log" 2>&1

echo ""
echo "All toy Bayesian partitions done."
