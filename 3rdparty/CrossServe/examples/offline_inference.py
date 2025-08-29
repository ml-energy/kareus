import argparse
from cfuser.config.args import ServerArgs
from cfuser.engine.runtime import CServeEngine
from cfuser.config import InputConfig
from copy import deepcopy


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_config = ServerArgs.from_cli_args(args)

    engine = CServeEngine(server_config)

    input_config = InputConfig(
        prompt=["A beautiful sunset over mountains"] * 1,
        height=2048,
        width=2048,
        num_inference_steps=3,
        batch_size=1,
        output_type="latent",
    )
    # for i in range(3):
    #     engine.add_request(input_config)

    import time

    start_time = time.time()
    engine.generate(input_config)
    end_time = time.time()
    print(f"time: {end_time - start_time}s")


"""
# non-scaling efficiency
python3 examples/offline_inference.py --nnodes 1 --nproc_per_node 4 \
--master_addr 127.0.0.1 --master_port 1037 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--schedule_logic "naive" \
--ulysses_degree 4 --ring_degree 1 2>&1 | tee log.log

# scaling efficiency
python3 examples/offline_inference.py --nnodes 1 --nproc_per_node 4 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--ulysses_degree 2 --ring_degree 2 2>&1 | tee log.log

# disaggregated efficiency
python3 examples/offline_inference.py --nnodes 1 --nproc_per_node 4 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "disaggregated_scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--ulysses_degree 2 --ring_degree 2 2>&1 | tee log.log

"""

if __name__ == "__main__":
    main()
