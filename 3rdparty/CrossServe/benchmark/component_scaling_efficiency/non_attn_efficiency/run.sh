#!/bin/bash

bs_list=(1 2 4 8 16 32)
seq_len_list=(8192 16384 32768 65536 1024 2048 4096)
world_size_list=(1 2 4 8)

# rm -rf log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json

# Function to run a command and retry if it fails
run_test() {
    local max_retries=3
    local attempt=0

    until "$@"; do
        exit_code=$?
        ((attempt++))
        echo "Command failed with exit code ${exit_code}. Attempt ${attempt}/${max_retries}."
        if [ ${attempt} -ge ${max_retries} ]; then
            echo "Max retries reached. Moving on..."
            break
        fi
        sleep 1
    done
}

for bs in "${bs_list[@]}"; do
    for seq_len in "${seq_len_list[@]}"; do
        for world_size in ${world_size_list[@]}; do
            run_test python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py \
                --batch_size "$bs" --seq_len "$seq_len" --parallel_degree $world_size --warmup_steps 100 --repeat 40 --logging
            pkill -f test_non_attn.py
        done

        # # Example for a single GPU run, run in background if needed.
        # CUDA_VISIBLE_DEVICES=0 run_test python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py \
        #     --batch_size "$bs" --seq_len "$seq_len" --parallel_degree 1 --warmup_steps 80 --repeat 40 --logging &
        # CUDA_VISIBLE_DEVICES=2,3 run_test python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py \
        #     --batch_size "$bs" --seq_len "$seq_len" --parallel_degree 2 --warmup_steps 80 --repeat 40 --logging &
        # CUDA_VISIBLE_DEVICES=4,5,6,7 run_test python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py \
        #     --batch_size "$bs" --seq_len "$seq_len" --parallel_degree 4 --warmup_steps 80 --repeat 40 --logging

        # pkill -f test_non_attn.py

        # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 run_test python3 benchmark/component_scaling_efficiency/non_attn_efficiency/test_non_attn.py \
        #     --batch_size "$bs" --seq_len "$seq_len" --parallel_degree 8 --warmup_steps 80 --repeat 40 --logging

        # pkill -f test_non_attn.py
    done
done
