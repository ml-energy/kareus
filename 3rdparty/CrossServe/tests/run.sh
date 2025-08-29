#!/bin/bash

# bash tests/run.sh 2>&1 | tee log.log

OUTPUT_JSON="log/tests/test_overlap_perf_correctness.json"

## without nsys
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --output_json $OUTPUT_JSON

CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 8 --output_json $OUTPUT_JSON

python tests/test_overlap_perf_correctness.py --ulysses_degree 8 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --output_json $OUTPUT_JSON

## no stream
CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --no_stream --output_json $OUTPUT_JSON

CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 8 --no_stream --output_json $OUTPUT_JSON

python tests/test_overlap_perf_correctness.py --ulysses_degree 8 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --no_stream --output_json $OUTPUT_JSON


# # with stream
# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --force-overwrite true -w true -s cpu  --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u4_r1_h1024_w1024 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8

# CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --force-overwrite true -w true -s cpu  --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u4_r1_h512_w512 python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 8

# nsys profile --force-overwrite true -w true -s cpu --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u8_r1_h1024_w1024 python tests/test_overlap_perf_correctness.py --ulysses_degree 8 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8

# # no stream
# CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile --force-overwrite true -w true -s cpu  --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u4_r1_h1024_w1024_no_stream python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --no_stream

# CUDA_VISIBLE_DEVICES=4,5,6,7 nsys profile --force-overwrite true -w true -s cpu  --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u4_r1_h512_w512_no_stream python tests/test_overlap_perf_correctness.py --ulysses_degree 4 --ring_degree 1 --height 512 --width 512 --num_inference_steps 8 --no_stream

# nsys profile --force-overwrite true -w true -s cpu --capture-range=cudaProfilerApi -o log/tests/test_overlap_perf_correctness_bs1_u8_r1_h1024_w1024_no_stream python tests/test_overlap_perf_correctness.py --ulysses_degree 8 --ring_degree 1 --height 1024 --width 1024 --num_inference_steps 8 --no_stream
