#!/bin/bash

GPU1=6
GPU2=7

# Test frequencies from 1700 to 1200 (step -100) and sm_num from 1 to 20
for frequency in 1900 1800 1700 1600 1500 1400 1300 1200; do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i $GPU1,$GPU2 --lock-gpu-clocks=${frequency},${frequency}
    
    for sm_num in {2..30..2}; do
        echo "  Testing sm_num: ${sm_num}"
        
        # Set NCCL channels
        export NCCL_MIN_NCHANNELS=${sm_num}
        export NCCL_MAX_NCHANNELS=${sm_num}
        
        # Create unique output filename
        output_file="output_${frequency}.log"
        
        # Run the test
        CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 python -u overlap_test_attn_nccl.py --frequency ${frequency} --sm_num ${sm_num} > ${output_file} 2>&1
        
        echo "    Completed: ${output_file}"
    done
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
