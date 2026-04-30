#!/usr/bin/env bash
# Bayesian-optimization driver for the artifact 16-GPU configs.
# Mirrors tests/bayesian/test_all_configs.sh: runs all evaluation configs on
# 8 GPUs with parallelism.
#   Phase 1: TP-only tests          (8 GPUs, sequential)
#   Phase 2: CP+TP TP-side parts    (4 GPUs each, 2 in parallel)
#   Phase 3: CP+TP CP-side parts    (2 GPUs each, 4 in parallel)
# Excludes: Llama 3.2 3B TP8 mbs=8/seq=8192 and mbs=16/seq=4096.
#
# This script does NOT itself need 16 GPUs — partition profilers run on small
# GPU subsets.  Run it once on a node with 8 GPUs visible before invoking
# tests/artifact/run_kareus.sh.
#
# Env var CONFIG_MODE (default full):
#   full   - full 3-phase sweep over all evaluation configs
#   single - only profile Llama 3.2 3B CP=1 TP=8 MBS=8 SEQ=4096
#            (matches run_kareus.sh CONFIG_MODE=single)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
[[ -f "${SCRIPT_DIR}/env.sh" ]] && source "${SCRIPT_DIR}/env.sh"
BAYESIAN_DIR="${SCRIPT_DIR}/../bayesian"
cd "$BAYESIAN_DIR"

CONFIG_MODE="${CONFIG_MODE:-full}"
case "${CONFIG_MODE}" in
    full|single) ;;
    *) echo "ERROR: CONFIG_MODE must be 'full' or 'single', got '${CONFIG_MODE}'" >&2; exit 1 ;;
esac

TP8_PARTS=(fwd_attn fwd_mlp)
CPTP_TP_PARTS=(fwd_qkv_ar fwd_ao_ar fwd_mlp bwd_qkv_ar bwd_o_ar bwd_mlp)
CPTP_CP_PARTS=(fwd_qkv_ag fwd_ao_ag bwd_qkv_rs bwd_a_rs bwd_a_ag bwd_o_ag)

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

run_tp_only() {
    local gpus=$1 model=$2 tp=$3 bs=$4 seq=$5
    local tag="${model}_tp${tp}_bs${bs}_seq${seq}"
    echo ">>> TP-only: ${tag}"
    for part in "${TP8_PARTS[@]}"; do
        local log_dir="logs/${model}/cp1-tp${tp}-bs${bs}-seq${seq}/${part}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -u partitions/${part}/bo_search.py \
            -m "$model" -w "$tp" -tp "$tp" -cp 1 -b "$bs" -s "$seq" \
            |& tee "${log_dir}/bo_${tag}.log"
    done
}

run_cptp_tp() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_TP_PARTS[@]}"; do
        local log_dir="logs/${model}/cp${cp}-tp${tp}-bs${bs}-seq${seq}/${part}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -u partitions/${part}/bo_search.py \
            -m "$model" -w "$tp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            |& tee "${log_dir}/bo_${tag}.log"
    done
}

run_cptp_cp() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS[@]}"; do
        local log_dir="logs/${model}/cp${cp}-tp${tp}-bs${bs}-seq${seq}/${part}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -u partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            |& tee "${log_dir}/bo_${tag}.log"
    done
}

CPTP_CP_PARTS_A=(bwd_a_rs fwd_ao_ag bwd_qkv_rs)
CPTP_CP_PARTS_B=(fwd_qkv_ag bwd_a_ag bwd_o_ag)

run_cptp_cp_a() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS_A[@]}"; do
        local log_dir="logs/${model}/cp${cp}-tp${tp}-bs${bs}-seq${seq}/${part}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -u partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            |& tee "${log_dir}/bo_${tag}.log"
    done
}

run_cptp_cp_b() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS_B[@]}"; do
        local log_dir="logs/${model}/cp${cp}-tp${tp}-bs${bs}-seq${seq}/${part}"
        mkdir -p "$log_dir"
        CUDA_VISIBLE_DEVICES=$gpus python -u partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            |& tee "${log_dir}/bo_${tag}.log"
    done
}

run_nonpartition() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    local log_dir="logs/${model}/cp${cp}-tp${tp}-bs${bs}-seq${seq}/nonpartition"
    mkdir -p "$log_dir"
    echo ">>> nonpartition: ${tag}  GPUs=${gpus}"
    CUDA_VISIBLE_DEVICES=$gpus python -u nonpartition/profile_nonpartition.py \
        -m "$model" -w "$tp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
        |& tee "${log_dir}/nonpartition_${tag}.log"
}

###############################################################################
# CONFIG_MODE=single — only Llama 3.2 3B CP=1 TP=8 MBS=8 SEQ=4096
###############################################################################
if [[ "${CONFIG_MODE}" == "single" ]]; then
    echo "===== CONFIG_MODE=single: Llama 3.2 3B CP=1 TP=8 MBS=8 SEQ=4096 ====="
    # run_tp_only 0,1,2,3,4,5,6,7 llama3.2_3b 8 8 4096
    run_nonpartition 0,1,2,3,4,5,6,7 llama3.2_3b 8 1 8 4096
    echo ""
    echo "Single-config bayesian profiling done."
    exit 0
fi

###############################################################################
# Phase 1 — TP-only (8 GPUs, sequential)
###############################################################################
echo "===== Phase 1: TP-only (8 GPUs) ====="
run_tp_only 0,1,2,3,4,5,6,7 llama3.2_3b 8 8  4096
run_nonpartition 0,1,2,3,4,5,6,7 llama3.2_3b 8 1 8  4096
run_tp_only 0,1,2,3,4,5,6,7 qwen3_1.7b  8 8  4096
run_nonpartition 0,1,2,3,4,5,6,7 qwen3_1.7b  8 1 8  4096
run_tp_only 0,1,2,3,4,5,6,7 qwen3_1.7b  8 8  8192
run_nonpartition 0,1,2,3,4,5,6,7 qwen3_1.7b  8 1 8  8192
run_tp_only 0,1,2,3,4,5,6,7 qwen3_1.7b  8 16 4096
run_nonpartition 0,1,2,3,4,5,6,7 qwen3_1.7b  8 1 16 4096

###############################################################################
# Phase 2 — CP+TP TP-side partitions (4 GPUs each, 2 in parallel)
###############################################################################
echo "===== Phase 2: CP+TP TP partitions + nonpartition (2×4 GPUs) ====="
pids=()
run_cptp_tp 0,1,2,3 llama3.2_3b 4 2 8  4096 & pids+=($!)
run_cptp_tp 4,5,6,7 llama3.2_3b 4 2 8  8192 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_nonpartition 0,1,2,3 llama3.2_3b 4 2 8  4096 & pids+=($!)
run_nonpartition 4,5,6,7 llama3.2_3b 4 2 8  8192 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_cptp_tp 0,1,2,3 llama3.2_3b 4 2 16 4096 & pids+=($!)
run_cptp_tp 4,5,6,7 qwen3_1.7b  4 2 8  4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_nonpartition 0,1,2,3 llama3.2_3b 4 2 16 4096 & pids+=($!)
run_nonpartition 4,5,6,7 qwen3_1.7b  4 2 8  4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_cptp_tp 0,1,2,3 qwen3_1.7b  4 2 8  8192 & pids+=($!)
run_cptp_tp 4,5,6,7 qwen3_1.7b  4 2 16 4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_nonpartition 0,1,2,3 qwen3_1.7b  4 2 8  8192 & pids+=($!)
run_nonpartition 4,5,6,7 qwen3_1.7b  4 2 16 4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

###############################################################################
# Phase 3 — CP+TP CP-side partitions (2 GPUs each, 4 in parallel)
###############################################################################
echo "===== Phase 3: CP+TP CP partitions (4×2 GPUs) ====="
run_cptp_cp 0,1 llama3.2_3b 4 2 8  4096 & pids+=($!)
run_cptp_cp 2,3 llama3.2_3b 4 2 8  8192 & pids+=($!)
run_cptp_cp 4,5 llama3.2_3b 4 2 16 4096 & pids+=($!)
run_cptp_cp 6,7 qwen3_1.7b  4 2 8  4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_cptp_cp_a 0,1 qwen3_1.7b  4 2 8  8192 & pids+=($!)
run_cptp_cp_b 2,3 qwen3_1.7b  4 2 8  8192 & pids+=($!)
run_cptp_cp_a 4,5 qwen3_1.7b  4 2 16 4096 & pids+=($!)
run_cptp_cp_b 6,7 qwen3_1.7b  4 2 16 4096 & pids+=($!)
wait_all "${pids[@]}"

echo ""
echo "All evaluation configs done."
