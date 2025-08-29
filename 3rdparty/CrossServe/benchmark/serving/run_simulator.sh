# !/bin/bash

# bash benchmark/serving/run_simulator.sh 2>&1 | tee log.log

arrival_rate=0.3
duration=100
seed=0
scaling_factor=15

python benchmark/serving/serving_simulator.py --schedule_logic naive --arrival_rate $arrival_rate --duration $duration --seed $seed --scaling_factor $scaling_factor

python benchmark/serving/serving_simulator.py --schedule_logic scaling_efficient --arrival_rate $arrival_rate --duration $duration --seed $seed --scaling_factor $scaling_factor

python benchmark/serving/serving_simulator.py --schedule_logic disaggregated_scaling_efficient --arrival_rate $arrival_rate --duration $duration --seed $seed --scaling_factor $scaling_factor

# python benchmark/serving/serving_simulator.py --schedule_logic disaggregated_scaling_efficient --arrival_rate 0.4 --duration 30 --seed 0 --scaling_factor 10 2>&1 | tee log.log
