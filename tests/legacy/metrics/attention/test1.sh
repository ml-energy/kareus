nvidia-smi -i 2,3 --lock-gpu-clocks=900,900
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 900 > output_900_8.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 900 >> output_900_8.log 2>&1

nvidia-smi -i 2,3 --lock-gpu-clocks=1350,1350
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1350 > output_1350_8.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1350 >> output_1350_8.log 2>&1

nvidia-smi -i 2,3 --lock-gpu-clocks=1250,1250
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1250 > output_1250_8.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1250 >> output_1250_8.log 2>&1

nvidia-smi -i 2,3 --lock-gpu-clocks=1150,1150
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1150 > output_1150_8.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1150 >> output_1150_8.log 2>&1

nvidia-smi -i 2,3 --lock-gpu-clocks=1050,1050
CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1050 > output_1050_8.log 2>&1
CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1050 >> output_1050_8.log 2>&1

# CUDA_VISIBLE_DEVICES=2,3 python -u profile_metrics_attn.py --batch_size 8 --frequency 1400
# CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1400 >> output_1400_8.log 2>&1

# nvidia-smi -i 2,3 --lock-gpu-clocks=1300,1300
# CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1300 > output_1300_8.log 2>&1

# nvidia-smi -i 2,3 --lock-gpu-clocks=1200,1200
# CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1200 > output_1200_8.log 2>&1
# CUDA_VISIBLE_DEVICES=2,3 python -u profile_metrics_attn.py --batch_size 8 --frequency 1200 >> output_1200_8.log 2>&1
# CUDA_VISIBLE_DEVICES=2,3 python -u test_attn_baseline.py --batch_size 8 --frequency 1200 >> output_1200_8.log 2>&1

# nvidia-smi -i 2,3 --lock-gpu-clocks=1000,1000
# CUDA_VISIBLE_DEVICES=2,3 python -u overlap_test_attn.py --batch_size 8 --frequency 1000 > output_1000_8.log 2>&1

