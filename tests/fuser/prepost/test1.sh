#!/bin/bash

GPU1=0
GPU2=1

# Test frequencies from 1700 to 1200 (step -100) and sm_num from 1 to 20
for frequency in 1700 1600 1500 1400 1300 1200 1100 1000; do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i ${GPU1},${GPU2} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_${frequency}.log"
    
    # Run the test
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_postprocess.py --frequency ${frequency} > ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python -u profile_postprocess_backward.py --frequency ${frequency} > ${output_file} 2>&1
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
