for frequency in $(seq 1740 -15 210); do
    nvidia-smi -i 0,1,2,3 --lock-gpu-clocks=${frequency},${frequency}
    python megatron_gpt_pretraining.py
done

# ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_1b/perseus_results_70/freqs_pipeline_01346.py"}' uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port 7787