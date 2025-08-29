#! /bin/bash

python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config.yml
python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config.yml --compile

# create your own config file
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_1.yml
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_2.yml
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_4.yml
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_8.yml

# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_2.yml --compile
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_4.yml --compile
# python benchmark/batching_scaling_benefits/launcher.py --config benchmark/batching_scaling_benefits/config_8.yml --compile
