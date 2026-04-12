# nvidia-smi -i 0,1 --lock-gpu-clocks=1400,1400
# CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 16 --frequency 1400 > output_1400_16.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 16 --frequency 1400 >> output_1400_16.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1 python -u profile_metrics_mlp.py --batch_size 16 --frequency 1400 >> output_1400_16.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=1350,1350
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 1350 > output_1350_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 1350 >> output_1350_8.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=1250,1250
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 1250 > output_1250_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 1250 >> output_1250_8.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=1150,1150
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 1150 > output_1150_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 1150 >> output_1150_8.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=900,900
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 900 > output_900_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 900 >> output_900_8.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=1050,1050
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 1050 > output_1050_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 1050 >> output_1050_8.log 2>&1

nvidia-smi -i 0,1 --lock-gpu-clocks=950,950
CUDA_VISIBLE_DEVICES=0,1 python -u overlap_test_mlp.py --batch_size 8 --frequency 950 > output_950_8.log 2>&1
CUDA_VISIBLE_DEVICES=0,1 python -u test_mlp_baseline.py --batch_size 8 --frequency 950 >> output_950_8.log 2>&1