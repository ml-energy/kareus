#!/bin/bash

GPU1=0
GPU2=1
# GPU3=2
# GPU4=3

for frequency in $(seq 1740 -60 1320); do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    # nvidia-smi -i ${GPU1},${GPU2},${GPU3},${GPU4} --lock-gpu-clocks=${frequency},${frequency}
    nvidia-smi -i ${GPU1},${GPU2} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_${frequency}.log"
    
    # Run the test
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_preprocess.py --frequency ${frequency}
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_preprocess_backward.py --frequency ${frequency}
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_loss.py --frequency ${frequency}
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_postprocess.py --frequency ${frequency}
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_postprocess_backward.py --frequency ${frequency}
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
