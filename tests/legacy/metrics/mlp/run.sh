#!/bin/bash

GPU1=0
GPU2=1

# python profile_metrics_mlp.py

for frequency in 1500 1300 1100; do
    echo "Testing frequency: ${frequency}"
    
    nvidia-smi -i ${GPU1},${GPU2} --lock-gpu-clocks=${frequency},${frequency}
    
    output_file="output_${frequency}.log"
    
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u overlap_test_mlp.py --frequency ${frequency} > ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_metrics_mlp.py --frequency ${frequency} >> ${output_file} 2>&1
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
