#!/bin/bash

GPU1=0
GPU2=1
GPU3=2
GPU4=3
GPU5=4
GPU6=5
GPU7=6
GPU8=7

# CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_transformer_layer_fuser.py > output_transformer_layer.log 2>&1
# CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_transformer_layer_backward.py > output_transformer_layer_backward.log 2>&1

CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_attention_fuser.py > output_attention.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_mlp_fuser.py > output_mlp.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_attention_fuser_backward.py > output_attention_backward.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_mlp_fuser_backward.py > output_mlp_backward.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_attention_fuser.py -b 2 > output_attention.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_mlp_fuser.py -b 2 > output_mlp.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_attention_fuser_backward.py -b 2 > output_attention_backward.log 2>&1
CUDA_VISIBLE_DEVICES=${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} python -u profile_mlp_fuser_backward.py -b 2 > output_mlp_backward.log 2>&1

nvidia-smi -i ${GPU1},${GPU2},${GPU3},${GPU4},${GPU5},${GPU6},${GPU7},${GPU8} --reset-gpu-clocks