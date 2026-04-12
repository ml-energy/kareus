#!/bin/bash

GPU1=4
GPU2=5

for frequency in $(seq 1410 -30 900); do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i ${GPU1},${GPU2} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_bwd_${frequency}.log"
    
    # Run the test
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u overlap_test_attn_backward.py --frequency ${frequency} > ${output_file} 2>&1
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
