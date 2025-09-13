#!/bin/bash

GPU1=4
GPU2=5
GPU3=6
GPU4=7

for frequency in $(seq 1140 -30 900); do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i ${GPU1},${GPU2},${GPU3},${GPU4} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_${frequency}.log"
    
    # Run the test
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4} python -u profile_preprocess.py --frequency ${frequency} > ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4} python -u profile_preprocess_backward.py --frequency ${frequency} >> ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4} python -u profile_loss.py --frequency ${frequency} >> ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4} python -u profile_postprocess.py --frequency ${frequency} >> ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4} python -u profile_postprocess_backward.py --frequency ${frequency} >> ${output_file} 2>&1
    
    echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
