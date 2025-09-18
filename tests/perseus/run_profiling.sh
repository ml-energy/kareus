for frequency in $(seq 1740 -60 900); do
    nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=${frequency},${frequency}
    python megatron_gpt_pretraining.py
done

nvidia-smi -i 0,1,2,3 --reset-gpu-clocks

mkdir -p nemo_experiments/megatron_llama_3_2_1b/profiling
mv nemo_experiments/megatron_llama_3_2_1b/* nemo_experiments/megatron_llama_3_2_1b/profiling/

# ZEUS_PFO_SCHEDULER=PointSolution3D ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_3b/perseus_results/freqs_pipeline_00390.py"}' uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port 7787