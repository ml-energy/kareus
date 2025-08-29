#!/bin/bash

# rm -rf log/benchmark/component_scaling_efficiency/e2e_scaling_efficiency/e2e_scaling_efficiency.json

CUDA_VISIBLE_DEVICES=0 python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 1 --repeat 4 --logging &
CUDA_VISIBLE_DEVICES=2,3 python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 2 --repeat 4 --logging

pkill -f test_e2e.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 4 --repeat 4 --logging

pkill -f test_e2e.py

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmark/component_scaling_efficiency/e2e_scaling_efficiency/test_e2e.py -g 8 --repeat 4 --logging

pkill -f test_e2e.py
