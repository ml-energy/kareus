#!/bin/bash

GPU1=0
GPU2=1
GPU3=2
GPU4=3
GPU5=4
GPU6=5
GPU7=6
GPU8=7

for frequency in $(seq 1410 -30 900); do
    echo "Testing frequency: ${frequency}"
    
    # Lock GPU clocks to current frequency
    nvidia-smi -i ${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} --lock-gpu-clocks=${frequency},${frequency}
    
    # Create unique output filename
    output_file="output_${frequency}.log"
    
    # Run the test
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess.py --frequency ${frequency} > ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess_backward.py --frequency ${frequency} >> ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_loss.py --frequency ${frequency} >> ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess.py --frequency ${frequency} >> ${output_file} 2>&1
    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess_backward.py --frequency ${frequency} >> ${output_file} 2>&1

    # output_file="output_${frequency}_8_8192.log"

    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess.py -b 8 -s 8192 --frequency ${frequency} > ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess_backward.py -b 8 -s 8192 --frequency ${frequency} >> ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_loss.py -b 8 -s 8192 --frequency ${frequency} >> ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess.py -b 8 -s 8192 --frequency ${frequency} >> ${output_file} 2>&1
    # CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess_backward.py -b 8 -s 8192 --frequency ${frequency} >> ${output_file} 2>&1
    
    # echo "    Completed: ${output_file}"
    
    echo "Completed frequency: ${frequency}"
    echo "----------------------------------------"
done

echo "All tests completed!"
