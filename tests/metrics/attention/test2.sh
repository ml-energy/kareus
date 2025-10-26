# nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1410,1410
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u profile_metrics_attn.py --frequency 1410 > profile_1410.log 2>&1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u overlap_test_attn.py --frequency 1410 > overlap_1410.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1200,1200
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u profile_metrics_attn.py --frequency 1200 > profile_1200.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u overlap_test_attn.py --frequency 1200 > overlap_1200.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1110,1110
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u profile_metrics_attn.py --frequency 1110 > profile_1110.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u overlap_test_attn.py --frequency 1110 > overlap_1110.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1020,1020
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u profile_metrics_attn.py --frequency 1020 > profile_1020.log 2>&1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u overlap_test_attn.py --frequency 1020 > overlap_1020.log 2>&1