#!/bin/bash

GPU1=0
GPU2=1
GPU3=2
GPU4=3
GPU5=4
GPU6=5
GPU7=6
GPU8=7

CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess.py -b 8 -s 4096 > output_preprocess.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_preprocess_backward.py -b 8 -s 4096 > output_preprocess_backward.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_loss.py -b 8 -s 4096 > output_loss.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess.py -b 8 -s 4096 > output_postprocess.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_postprocess_backward.py -b 8 -s 4096 > output_postprocess_backward.log 2>&1

nvidia-smi -i ${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} --reset-gpu-clocks