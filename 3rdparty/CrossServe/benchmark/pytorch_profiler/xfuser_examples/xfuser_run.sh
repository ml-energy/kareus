# !/bin/bash

# RESOLUTIONS=(360 512 720 1024 2048 4096)
RESOLUTIONS=(360 512 720 1024)
ULYSSES_DEGREES=(1 2 4)
GPU_DEVICES="0,1,2,3"
MASTER_PORT=1049
MODEL="black-forest-labs/FLUX.1-dev"
PROMPT="hello world"
NUM_STEPS=3
RING_DEGREES=(1 2 4)

for res in "${RESOLUTIONS[@]}"; do
    for ulysses in "${ULYSSES_DEGREES[@]}"; do
        for ring in "${RING_DEGREES[@]}"; do
            echo "Running with resolution ${res}x${res}, Ulysses degree ${ulysses}, Ring degree ${ring}"

            total_gpus=$((ring * ulysses))
            if [ ${total_gpus} -gt 4 ]; then
                continue
            fi

            NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun \
                --nproc_per_node=${total_gpus} \
                --master_port=${MASTER_PORT} \
                benchmark/pytorch_profiler/xfuser_examples/xfuser_examples.py \
                --prompt "${PROMPT}" \
                --output_type 'latent' \
                --model "${MODEL}" \
                --height ${res} \
                --width ${res} \
                --ulysses_degree ${ulysses} \
                --ring_degree ${ring} \
                --num_inference_steps ${NUM_STEPS}

            # Wait a bit between runs to let GPUs cool down
            sleep 2
        done
    done
done


# !/bin/bash

# RESOLUTIONS=(360 512 720 1024 2048 4096)
# RESOLUTIONS=(1024 720 512 360)
# ULYSSES_DEGREES=(1 2 4)
# GPU_DEVICES="4,5,6,7"
# MASTER_PORT=1037
# MODEL="black-forest-labs/FLUX.1-dev"
# PROMPT="hello world"
# NUM_STEPS=3
# RING_DEGREES=(1 2 4)

# for res in "${RESOLUTIONS[@]}"; do
#     for ulysses in "${ULYSSES_DEGREES[@]}"; do
#         for ring in "${RING_DEGREES[@]}"; do
#             echo "Running with resolution ${res}x${res}, Ulysses degree ${ulysses}, Ring degree ${ring}"

#             total_gpus=$((ring * ulysses))
#             if [ ${total_gpus} -gt 4 ]; then
#                 continue
#             fi

#             NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=${GPU_DEVICES} torchrun \
#                 --nproc_per_node=${total_gpus} \
#                 --master_port=${MASTER_PORT} \
#                 benchmark/pytorch_profiler/xfuser_examples/xfuser_examples.py \
#                 --prompt "${PROMPT}" \
#                 --output_type 'latent' \
#                 --model "${MODEL}" \
#                 --height ${res} \
#                 --width ${res} \
#                 --ulysses_degree ${ulysses} \
#                 --ring_degree ${ring} \
#                 --num_inference_steps ${NUM_STEPS}

#             # Wait a bit between runs to let GPUs cool down
#             sleep 2
#         done
#     done
# done



# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1039 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 720 --width 720 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3


# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=1048 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 2048 --width 2048 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3


# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1058 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 2048 --width 2048 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3


# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1042 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --ulysses_degree 4 --ring_degree 1 --num_inference_steps 3

# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1078 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 1024 --width 1024 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3


# NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1057 examples/xfuser_examples.py --prompt 'hello world' --output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
# --height 720 --width 720 --ulysses_degree 2 --ring_degree 1 --num_inference_steps 3
