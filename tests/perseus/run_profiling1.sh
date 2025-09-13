for frequency in $(seq 1410 -30 900); do
    nvidia-smi -i 0,1,2,3,4,5,6,7 --lock-gpu-clocks=${frequency},${frequency}
    bash ./run1.sh
    mkdir -p nemo_experiments/megatron_qwen3_1p7b/${frequency}
    mv nemo_experiments/megatron_qwen3_1p7b/timers nemo_experiments/megatron_qwen3_1p7b/${frequency}/
    mv nemo_experiments/megatron_qwen3_1p7b/*.txt nemo_experiments/megatron_qwen3_1p7b/${frequency}/
done

nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

mkdir -p nemo_experiments/megatron_qwen3_1p7b/profiling/node1
mv nemo_experiments/megatron_qwen3_1p7b/* nemo_experiments/megatron_qwen3_1p7b/profiling/node1/
