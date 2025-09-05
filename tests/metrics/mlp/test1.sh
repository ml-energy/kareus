nvidia-smi -i 2,3 --lock-gpu-clocks=1200,1200
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_mlp.py --batch_size 16 --frequency 1200 > output_1200_16.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_mlp_baseline.py --batch_size 16 --frequency 1200 >> output_1200_16.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u profile_metrics_mlp.py --batch_size 16 --frequency 1200 >> output_1200_16.log 2>&1
