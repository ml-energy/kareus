export MASTER_ADDR=172.31.44.236

ZEUS_PFO_SCHEDULER=PointSolution3D ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "nemo_experiments/megatron_llama_3_2_1b/freqs_pipeline_01491.py"}' uvicorn zeus.optimizer.pipeline_frequency.server.router:app --host 0.0.0.0 --port 7787