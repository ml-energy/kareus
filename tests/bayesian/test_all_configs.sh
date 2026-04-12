#!/usr/bin/env bash
# Run all evaluation configs on 8 GPUs with parallelism.
#   Phase 1: TP-only tests          (8 GPUs, sequential)
#   Phase 2: CP+TP TP-side parts    (4 GPUs each, 2 in parallel)
#   Phase 3: CP+TP CP-side parts    (2 GPUs each, 4 in parallel)
# Excludes: Llama 3.2 3B TP8 mbs=8/seq=8192 and mbs=16/seq=4096.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TP8_PARTS=(fwd_attn fwd_mlp bwd_attn bwd_mlp)
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
    local model=$1 tp=$2 bs=$3 seq=$4
    local tag="${model}_tp${tp}_bs${bs}_seq${seq}"
    echo ">>> TP-only: ${tag}"
    for part in "${TP8_PARTS[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python partitions/${part}/bo_search.py \
            -m "$model" -w "$tp" -tp "$tp" -cp 1 -b "$bs" -s "$seq" \
            > "partitions/${part}/bo_${tag}.log" 2>&1
    done
}

run_cptp_tp() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_TP_PARTS[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpus python partitions/${part}/bo_search.py \
            -m "$model" -w "$tp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            > "partitions/${part}/bo_${tag}.log" 2>&1
    done
}

run_cptp_cp() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpus python partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            > "partitions/${part}/bo_${tag}.log" 2>&1
    done
}

CPTP_CP_PARTS_A=(bwd_a_rs fwd_ao_ag bwd_qkv_rs)
CPTP_CP_PARTS_B=(fwd_qkv_ag bwd_a_ag bwd_o_ag)

run_cptp_cp_a() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS_A[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpus python partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            > "partitions/${part}/bo_${tag}.log" 2>&1
    done
}

run_cptp_cp_b() {
    local gpus=$1 model=$2 tp=$3 cp=$4 bs=$5 seq=$6
    local tag="${model}_cp${cp}_tp${tp}_bs${bs}_seq${seq}"
    for part in "${CPTP_CP_PARTS_B[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpus python partitions/${part}/bo_search.py \
            -m "$model" -w "$cp" -tp "$tp" -cp "$cp" -b "$bs" -s "$seq" \
            > "partitions/${part}/bo_${tag}.log" 2>&1
    done
}

###############################################################################
# Phase 1 — TP-only (8 GPUs, sequential)
###############################################################################
echo "===== Phase 1: TP-only (8 GPUs) ====="
run_tp_only llama3.2_3b 8 8  4096
run_tp_only qwen3_1.7b  8 8  4096
run_tp_only qwen3_1.7b  8 8  8192
run_tp_only qwen3_1.7b  8 16 4096

###############################################################################
# Phase 2 — CP+TP TP-side partitions (4 GPUs each, 2 in parallel)
###############################################################################
echo "===== Phase 2: CP+TP TP partitions (2×4 GPUs) ====="
pids=()
run_cptp_tp 0,1,2,3 llama3.2_3b 4 2 8  4096 & pids+=($!)
run_cptp_tp 4,5,6,7 llama3.2_3b 4 2 8  8192 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_cptp_tp 0,1,2,3 llama3.2_3b 4 2 16 4096 & pids+=($!)
run_cptp_tp 4,5,6,7 qwen3_1.7b  4 2 8  4096 & pids+=($!)
wait_all "${pids[@]}"; pids=()

run_cptp_tp 0,1,2,3 qwen3_1.7b  4 2 8  8192 & pids+=($!)
run_cptp_tp 4,5,6,7 qwen3_1.7b  4 2 16 4096 & pids+=($!)
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
