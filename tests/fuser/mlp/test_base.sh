#!/bin/bash

GPU1=2
GPU2=3

# Test frequencies from 1700 to 1200 (step -100) and sm_num from 1 to 20
for frequency in 1400 1300 1200 1100 1000; do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i ${GPU1},${GPU2} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_baseline_${frequency}.log"
    
    # Run the test
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u test_mlp_baseline.py --frequency ${frequency} > ${output_file} 2>&1
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
