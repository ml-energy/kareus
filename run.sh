cd tests/fuser/
bash test_fwd_latency.sh

cargo run -- data/A100/profiles/llama3.2_profile.csv 

cd tests/perseus
# /workspaces/Kareus/tests/simple_test/data/my-gpt_text_document
# droupout?
nsys profile -c cudaProfilerApi -o megatron_llama1b -f true python megatron_gpt_pretraining.py
# nsys_profile: False

cd tests/kareus
nsys profile -c cudaProfilerApi -o dominok_llama1b -f true python kareus_gpt_pretraining.py
# nsys_profile: False

# Megatron baseline
cd tests/perseus
# enable_megatron_timers: true
bash run_profiling.sh
python profile_p2p.py
python generate_profile_csv.py
# update num_mbs, num_stages, p2p_power
python run_optimization.py

# Megatron + Perseus
# enable_megatron_timers: false
# enable_zeus_monitor: true
python megatron_gpt_pretraining.py
mv nemo_experiments/megatron_llama_3_2_1b/2025* nemo_experiments/megatron_llama_3_2_1b/baseline/
# enable_perseus_optimizer: true
# ZEUS_PFO_SCHEDULER=PointSolution3D ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_3b/perseus_results/freqs_pipeline_*.py"}' uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port 7787
python megatron_gpt_pretraining.py
mv nemo_experiments/megatron_llama_3_2_1b/2025* nemo_experiments/megatron_llama_3_2_1b/optimized/
python parse_results.py

# Kareus partition
cd tests/fuser/prepost
bash test.sh

cd tests/bayesian/ # update p2p power
bash run.sh

cd tests/kareus
python generate_profile_csv_from_bo.py
python run_optimization.py
# cp nemo_experiments/megatron_llama_3_2_3b/kareus_results/scheds_pipeline_*.py to /workspaces/Kareus/tests/kareus/conf/solution.py

# Megatron + Kareus
# enable_zeus_monitor: true
# enable_kareus_scheduler: true
# enable_perseus_optimizer: true
# ZEUS_PFO_SCHEDULER=PointSolution3D ZEUS_PFO_SCHEDULER_ARGS='{"solution_path": "/workspaces/Kareus/tests/kareus/nemo_experiments/megatron_llama_3_2_3b/kareus_results/freqs_pipeline_*.py"}' uvicorn zeus.optimizer.pipeline_frequency.server.router:app --port 7787
python kareus_gpt_pretraining.py
