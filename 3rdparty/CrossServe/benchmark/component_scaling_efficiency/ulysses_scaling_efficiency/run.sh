bs_list=(1 2 4 8 16 32)
seq_len_list=(8192 16384 32768 65536 1024 2048 4096)
world_size_list=(1 2 4 8)

# rm -rf log/benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/ulysses_scaling_efficiency.json
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

for bs in ${bs_list[@]}; do
    for seq_len in ${seq_len_list[@]}; do
        for world_size in ${world_size_list[@]}; do
            run_test python benchmark/component_scaling_efficiency/ulysses_scaling_efficiency/test_ulysses.py --bs $bs --seq_len $seq_len --world_size $world_size --logging
            pkill -f test_ulysses.py
        done
    done
done
