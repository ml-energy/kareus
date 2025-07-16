nvidia-smi -i 0,1,2,3 -pm 1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1200,1200
python -u overlap_test_attn.py --frequency 1200 > output.log 2>&1