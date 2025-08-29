bs_list=(1 2 4 8 16 32)
seq_len_list=(8192 16384 32768 65536 1024 2048 4096)
world_size_list=(1 2 4 8)

# rm -rf log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json

run_test() {
    local max_retries=2
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

for bs in ${bs_list[@]}; do
    for seq_len in ${seq_len_list[@]}; do
        # 8 GPUs in a node
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 1037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 1 --logging &
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 1038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 2 --ring_attn_world_size 1 --logging &
        CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port 1039 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 4 --ring_attn_world_size 1 --logging &
        CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 1040 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 8 --ring_attn_world_size 1 --logging

        pkill -f test_ring_attn.py

        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 2037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 2 --logging &
        CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 2038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 2 --ring_attn_world_size 2 --logging
        CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 --master_port 2038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 4 --ring_attn_world_size 2 --logging

        kill -9 $(pgrep -f test_ring_attn.py)

        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 3037 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 4 --logging &
        CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 3038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 2 --ring_attn_world_size 4 --logging

        kill -9 $(pgrep -f test_ring_attn.py)

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 3038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 8 --logging


        # # 4 GPUs in a node
        # CUDA_VISIBLE_DEVICES=0 run_test python benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 1 --logging &
        # CUDA_VISIBLE_DEVICES=2,3 run_test python benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 2 --logging

        # pkill -f test_ring_attn.py

        # CUDA_VISIBLE_DEVICES=0,1,2,3 run_test python benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size 4 --logging

        # pkill -f test_ring_attn.py

        # # CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 1038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 2 --ring_attn_world_size 1 --logging &
        # # CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port 1039 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 4 --ring_attn_world_size 1 --logging &
        # # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 2038 benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 2 --ring_attn_world_size 2 --logging

        # # kill -9 $(pgrep -f test_ring_attn.py)

        for world_size in ${world_size_list[@]}; do
            #random port
            master_port=$((RANDOM%10000+1000))
            run_test torchrun --nproc_per_node $world_size --master_port $master_port \
                benchmark/component_scaling_efficiency/ring_scaling_efficiency/test_ring_attn.py --bs $bs --seq_len $seq_len --ulysses_world_size 1 --ring_attn_world_size $world_size --logging
            pkill -f test_ring_attn.py
        done
    done
done
