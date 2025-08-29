#!/bin/bash

BS=2
HEIGHT=1024
WIDTH=1024

NPROC_PER_NODE=8

# for HEIGHT in 1024 2048 4096; do
#     for BS in 1 2 4 8; do
#         WIDTH=${HEIGHT}
#         echo "Running benchmarks with batch size ${BS} height ${HEIGHT} width ${WIDTH}"

#         # non-scaling efficiency
#         python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
#         --master_addr 127.0.0.1 --master_port 1037 \
#         --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
#         --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
#         --ulysses_degree 4 --ring_degree 1 --num_inference_steps 5 --logging 2>&1 | tee log1.log

#         python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 2 \
#         --master_addr 127.0.0.1 --master_port 1037 \
#         --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
#         --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
#         --ulysses_degree 2 --ring_degree 1 --num_inference_steps 5 --logging 2>&1 | tee log10.log

#         python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 1 \
#         --master_addr 127.0.0.1 --master_port 1037 \
#         --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
#         --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
#         --ulysses_degree 1 --ring_degree 1 --num_inference_steps 5 --logging 2>&1 | tee log11.log
#     done
# done

for BS in 1 2 4 8; do
    for HEIGHT in 1024 2048 4096; do
        WIDTH=${HEIGHT}
        # non-scaling efficiency
        python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        --master_addr 127.0.0.1 --master_port 1037 \
        --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
        --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
        --ulysses_degree 4 --ring_degree 1 --num_inference_steps 5 2>&1 --logging | tee log1.log

        # scaling efficiency
        python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        --master_addr 127.0.0.1 --master_port 1037 \
        --schedule_logic "scaling_efficient" \
        --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
        --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
        --ulysses_degree 2 --ring_degree 2 --num_inference_steps 5 2>&1 --logging | tee log2.log

        # disaggregated efficiency
        python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
        --master_addr 127.0.0.1 --master_port 1037 \
        --schedule_logic "disaggregated_scaling_efficient" \
        --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
        --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' \
        --ulysses_degree 2 --ring_degree 2 --num_inference_steps 5 2>&1 --logging | tee log3.log
    done
done

# export nsys_args="--force-overwrite true -w true -s cpu --python-backtrace=cuda --cudabacktrace=all --capture-range=cudaProfilerApi"
export nsys_args="--force-overwrite true -w true --capture-range=cudaProfilerApi"

# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_naive \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 2 --ring_degree 2 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent'

# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_scaling \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 2 --ring_degree 2 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' --schedule_logic "scaling_efficient"

# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_disaggregated_no_barrier \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 2 --ring_degree 2 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent' --schedule_logic "disaggregated_scaling_efficient"

# ### test naive 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_naive_parallel_1 \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 1 \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 1 --ring_degree 1 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent'

# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_naive_parallel_2 \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 2 \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 1 --ring_degree 2 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent'

# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile ${nsys_args} -o log/benchmark/disaggregated_offline_benefits/batch_${BS}_height_${HEIGHT}_width_${WIDTH}_naive_parallel_4 \
# python benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 8 \
# --master_addr 127.0.0.1 --master_port 1037 \
# --ulysses_degree 1 --ring_degree 4 --batch_size ${BS} --height ${HEIGHT} --width ${WIDTH} --num_inference_steps 5 --output_type 'latent'
