nvidia-smi -i 0,1,2,3 -pm 1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1700,1700
python -u overlap_test_attn.py --frequency 1700 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1600,1600
python -u overlap_test_attn.py --frequency 1600 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1500,1500
python -u overlap_test_attn.py --frequency 1500 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1400,1400
python -u overlap_test_attn.py --frequency 1400 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1300,1300
python -u overlap_test_attn.py --frequency 1300 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1200,1200
python -u overlap_test_attn.py --frequency 1200 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1000,1000
python -u overlap_test_attn.py --frequency 1000 > output.log 2>&1

nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=1100,1100
python -u overlap_test_attn.py --frequency 1100 > output.log 2>&1
