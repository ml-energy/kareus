import os
import yaml
import subprocess


def launch_benchmark(yaml_config: str, compile: bool = False):
    with open(yaml_config, "r") as f:
        config = yaml.safe_load(f)

    # Create log directory
    logdir = os.environ.get("BENCHMARK_LOG", "log/benchmark/batching_scaling_benefits")
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    gpu_configs = config["gpu_configs"]
    processes = []

    for config in gpu_configs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = config["gpus"]

        u = config["u"]
        r = config["r"]

        # Create log filename based on configuration
        compile_suffix = "_compiled" if compile else ""
        log_filename = f"u{u}_r{r}_gpus{config['gpus'].replace(',','_')}{compile_suffix}.log"
        log_path = os.path.join(logdir, log_filename)

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(u * r),
            (
                "benchmark/batching_scaling_benefits/benchmark.py"
                if os.path.exists("benchmark/batching_scaling_benefits/benchmark.py")
                else "benchmark.py"
            ),
            "--ulysses_degree",
            str(u),
            "--ring_degree",
            str(r),
        ]
        if compile:
            cmd.append("--use_compile")

        # Convert command list to string for logging
        cmd_str = " ".join(cmd)

        # Open log file and write initial information
        log_file = open(log_path, "w")
        log_file.write(f"Start process cmd: {cmd_str}\n")
        log_file.write(f"Environment Variables:\n")
        for k, v in env.items():
            log_file.write(f"    {k}: {v}\n")
        log_file.write("\n=== OUTPUT ===\n")

        print(f"Starting: Ulysses Degree: {u}, Ring Degree: {r} on GPUs: {config['gpus']}")
        print(f"Log file: {log_path}")

        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        processes.append((p, log_file))

    # Wait for all processes to complete
    for p, log_file in processes:
        p.wait()
        log_file.close()

        if p.returncode != 0:
            print(f"Process failed with return code {p.returncode}")
            print(f"Check logs in {log_file.name}")
        else:
            print(f"Process completed successfully. Logs written to {log_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compile", action="store_true")
    # @Wenxuan: I'd do this, but you can keep it as is
    # parser.add_argument("config", type=str, help="Path to the yaml config file")
    # parser.add_argument("--compile", "-c", action="store_true", help="Whether to use compiled kernels")

    args = parser.parse_args()

    config_path = args.config
    compile = True if args.compile else False

    launch_benchmark(config_path, compile=compile)
