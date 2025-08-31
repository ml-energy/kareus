#!/bin/bash

# CUDA_VISIBLE_DEVICES=6,7 python -u overlap_test_mlp.py > output_default_16.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6,7 python -u test_mlp_baseline.py >> output_default_16.log 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 python -u profile_metrics_mlp.py > output_default_16.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4,5 python -u overlap_test_mlp.py --batch_size 8 > output_default_8.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5 python -u test_mlp_baseline.py --batch_size 8 >> output_default_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 python -u profile_metrics_mlp.py --batch_size 8 > output_default_8.log 2>&1 &

nvidia-smi -i 2,3 --lock-gpu-clocks=1200,1200
# CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_mlp.py --frequency 1200 > output_1200_16.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 python -u test_mlp_baseline.py --frequency 1200 >> output_1200_16.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u profile_metrics_mlp.py --frequency 1200 > output_1200_16.log 2>&1 &

nvidia-smi -i 0,1 --lock-gpu-clocks=1200,1200
# CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 1200 > output_1200_8.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 1200 >> output_1200_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u profile_metrics_mlp.py --batch_size 8 --frequency 1200 > output_1200_8.log 2>&1 &