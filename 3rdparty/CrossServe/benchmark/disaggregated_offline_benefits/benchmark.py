import os
import time
import json
import torch
import argparse
from cfuser.config.args import ServerArgs
from cfuser.engine.runtime import CServeEngine
from cfuser.config import InputConfig
from copy import deepcopy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging", action="store_true", default=False)
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_config = ServerArgs.from_cli_args(args)

    engine = CServeEngine(server_config)

    input_config = InputConfig(
        prompt=["A beautiful sunset over mountains"] * args.batch_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        batch_size=args.batch_size,
        output_type=args.output_type,
    )

    # warmup
    for i in range(5):
        engine.add_request(input_config)
    engine.generate(input_config, send_done_packet=False)

    print("---------warmup done---------")

    torch.cuda.cudart().cudaProfilerStart()
    # input_config.num_inference_steps = 8
    # benchmark
    for i in range(3):
        engine.add_request(input_config)

    start_time = time.perf_counter()
    engine.generate(input_config)

    end_time = time.perf_counter()
    print(
        f"batchsize {args.batch_size} height {args.height} width {args.width} num_inference_steps {args.num_inference_steps} time: {end_time - start_time}s"
    )
    if args.logging:
        json_path = f"log/benchmark/disaggregated_offline_benefits/benchmark.json"
        to_save = {
            "bs": args.batch_size,
            "height": args.height,
            "width": args.width,
            "ulysses_degree": args.ulysses_degree,
            "ring_degree": args.ring_degree,
            "num_inference_steps": args.num_inference_steps,
            "time": end_time - start_time,
            "schedule_logic": args.schedule_logic,
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Load existing data
        existing_data = []
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing_data = json.load(f)

        # Append new data
        if not isinstance(existing_data, list):
            existing_data = [existing_data]
        existing_data.append(to_save)

        # Save updated data
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=2)

    torch.cuda.cudart().cudaProfilerStop()


"""
# non-scaling efficiency
python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 1 --height 1024 --width 1024 --num_inference_steps 5 --output_type 'latent' \
--ulysses_degree 4 --ring_degree 1 --num_inference_steps 5 2>&1 --logging | tee log1.log

# scaling efficiency
python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 1 --height 1024 --width 1024 --num_inference_steps 5 --output_type 'latent' \
--ulysses_degree 2 --ring_degree 2 --num_inference_steps 5 2>&1 --logging | tee log2.log

# disaggregated efficiency
python3 benchmark/disaggregated_offline_benefits/benchmark.py --nnodes 1 --nproc_per_node 8 \
--master_addr 127.0.0.1 --master_port 1037 \
--schedule_logic "disaggregated_scaling_efficient" \
--output_type 'latent' --model "black-forest-labs/FLUX.1-dev" \
--batch_size 1 --height 1024 --width 1024 --num_inference_steps 5 --output_type 'latent' \
--ulysses_degree 2 --ring_degree 2 --num_inference_steps 5 2>&1 --logging | tee log3.log
"""

if __name__ == "__main__":
    main()
