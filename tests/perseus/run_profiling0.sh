for frequency in $(seq 1410 -30 900); do
    nvidia-smi -i 0,1,2,3,4,5,6,7 --lock-gpu-clocks=${frequency},${frequency}
    bash ./run0.sh
    mv nemo_experiments/megatron_llama_3_2_1b/2025* nemo_experiments/megatron_llama_3_2_1b/${frequency}/
    mv nemo_experiments/megatron_llama_3_2_1b/timers/* nemo_experiments/megatron_llama_3_2_1b/${frequency}/timers
    mv nemo_experiments/megatron_llama_3_2_1b/*.txt nemo_experiments/megatron_llama_3_2_1b/${frequency}/
done

nvidia-smi -i 0,1,2,3,4,5,6,7 --reset-gpu-clocks

mkdir -p nemo_experiments/megatron_llama_3_2_1b/profiling/node0
mv nemo_experiments/megatron_llama_3_2_1b/* nemo_experiments/megatron_llama_3_2_1b/profiling/node0/
