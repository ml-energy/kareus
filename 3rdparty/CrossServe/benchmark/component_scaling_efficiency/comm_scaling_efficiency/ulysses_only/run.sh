bs_list=(1 2 4 8 16 32)
seq_len_list=(16384 32768 65536 1024 2048 4096 8192)
mlp_world_size_list=(2 4 8)  # Remove spaces around `=`

rm -rf log/benchmark/component_scaling_efficiency/comm_scaling_efficiency/ulysses_only/comm_scaling_efficiency.json

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
        for mlp_world_size in "${mlp_world_size_list[@]}"; do
            visible_devices=$(seq -s, 0 $((mlp_world_size-1)))  # Use `-s,` to separate devices by commas
            attn_world_sizes=()
            value=1
            while [ "$value" -le "$mlp_world_size" ]; do
                attn_world_sizes+=("$value")
                value=$((value * 2))
            done

            for attn_world_size in "${attn_world_sizes[@]}"; do
                # only test all to all once per mlp world size
                skip_a2a=False
                if [ "$attn_world_size" -ne "$mlp_world_size" ]; then
                    skip_a2a=True
                fi

                CUDA_VISIBLE_DEVICES=$visible_devices run_test python benchmark/component_scaling_efficiency/comm_scaling_efficiency/ulysses_only/test_ulysses.py \
                    --bs "$bs" --seq_len "$seq_len" --mlp_world_size "$mlp_world_size" --attn_world_size "$attn_world_size" --logging --skip_a2a "$skip_a2a"

                pkill -f test_ulysses.py  # More reliable than `kill -9 $(pgrep -f test_ulysses.py)`
            done
        done
    done
done
