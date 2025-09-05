nvidia-smi -i 0,1 --lock-gpu-clocks=1100,1100
# CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_attn.py --batch_size 8 --frequency 1100 > output_1100_8.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1 python -u test_attn_baseline.py --batch_size 8 --frequency 1100 >> output_1100_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u profile_metrics_attn.py --batch_size 8 --frequency 1100
