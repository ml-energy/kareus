#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile energy and time of the MLP fuser only.

Execution details:
- Distributed spawn over tensor-parallel world size.
- Uses ZeusMonitor to record time and GPU energy.
- Overlap scope: (-1, -1)
- SM configs: (None, None)
"""

import os
import sys
import time
import random
import argparse
import traceback

import torch
import torch.distributed as dist
import pynvml
import gc

# Make all required paths importable
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

ATTN_DIR = os.path.join(REPO_ROOT, "tests", "bayesian", "attention")
if ATTN_DIR not in sys.path:
    sys.path.append(ATTN_DIR)
MLP_DIR = os.path.join(REPO_ROOT, "tests", "bayesian", "mlp")
if MLP_DIR not in sys.path:
    sys.path.append(MLP_DIR)
FUSER_TESTS_DIR = os.path.join(REPO_ROOT, "tests", "fuser")
if FUSER_TESTS_DIR not in sys.path:
    sys.path.append(FUSER_TESTS_DIR)

from zeus.monitor import ZeusMonitor  # noqa: E402
from common_config import FuserTestConfig  # noqa: E402

# Ops for MLP fuser
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser  # noqa: E402
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp  # noqa: E402
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm  # noqa: E402
from kareus.transformer_engine.pytorch.ops.linear import Linear  # noqa: E402
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce  # noqa: E402
from kareus.megatron.core.extensions.ops import BiasSwigluOp  # noqa: E402


FREQ_VALUES = list(range(1410, 890, -30))


def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    if world_size <= 1:
        return None
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        print(f"[rank {rank}] initialized distributed world_size={world_size}")
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks)
    if rank == 0:
        print(f"Created TP group: {ranks}")
    return tp_group


def _set_gpu_frequency(target_freq_mhz, device_indices=None):
    """Best-effort GPU application clock setter via NVML."""
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(target_freq_mhz), int(target_freq_mhz))
    time.sleep(2)
    pynvml.nvmlShutdown()


class MLPFuserProfiler:
    """
    Build and profile the MLP fuser forward only.
    Measures average step time and energy across iterations.
    """

    def __init__(self, args: argparse.Namespace, rank: int, world_size: int) -> None:
        self.args = args
        self.rank = rank
        self.world_size = world_size

        assert world_size == args.tensor_parallel_size, \
            "world_size must equal --tensor_parallel_size"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.tp_group = init_distributed(rank, world_size)

        self.tensor_parallel_size = world_size
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len // args.context_parallel_size
        self.frequency = getattr(args, "frequency", "default")

        self.model_name = args.model_name
        self.context_parallel_size = args.context_parallel_size

        # Model config values
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE

        # Create MLP config
        self.mlp_config = FuserTestConfig.create_mlp_config(
            context_parallel_size=self.context_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
        )

        # Build MLP tensors (standalone)
        self.mlp_hidden_states = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        self.mlp_bias = None
        self.mlp_residual = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        self.mlp_allreduce_inputs = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        # MLP ops
        mlp_bda = BiasDropoutAddOp(
            dropout_prob=self.mlp_config.hidden_dropout, training=True
        )
        mlp_rms = RMSNorm(
            normalized_shape=self.hidden_size,
            eps=self.mlp_config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype,
        )
        fc1_hidden = (2 * self.ffn_hidden_size) // self.tensor_parallel_size
        mlp_fc1 = Linear(
            in_features=self.hidden_size,
            out_features=fc1_hidden,
            device=self.device,
            dtype=self.dtype,
            bias=False,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        mlp_swiglu = BiasSwigluOp(fp8_input_store=self.mlp_config.activation_func_fp8_input_store)
        fc2_hidden = self.ffn_hidden_size // self.tensor_parallel_size
        mlp_fc2 = Linear(
            in_features=fc2_hidden,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=False,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        mlp_allreduce = AllReduce(
            process_group=self.tp_group,
            async_op=True,
            backend="msccl",
            rank=self.rank,
            world_size=self.world_size,
            use_persistent_output=True,
            input_buffer=self.mlp_allreduce_inputs,
            tensor_size=[self.seq_length, self.batch_size, self.hidden_size],
            device=self.device,
            dtype=self.dtype,
        )
        self.mlp_comp_ops = [
            mlp_bda,
            mlp_rms,
            mlp_fc1,
            mlp_swiglu,
            mlp_fc2,
        ]
        self.mlp_comm_op = mlp_allreduce
        self.mlp_fuser = PartitionFuser(
            ops=self.mlp_comp_ops,
            comm_op_fwd=self.mlp_comm_op,
            fuse_ops=False,
            is_first_attn=False,
            is_last_mlp=True,
        )

        self.overlap_window = (-1, -1)
        self.sm_configs = (None, None)

    def _logs_dir(self) -> str:
        logs_dir = f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.args.seq_len}/{self.frequency}"
        if self.rank == 0:
            os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def _run_one_step(self):
        mlp_out, mlp_out_bias, mlp_out_residual, _mlp_allreduce = self.mlp_fuser(
            hidden_states=self.mlp_hidden_states,
            bias=self.mlp_bias,
            residual=self.mlp_residual,
            comm_input=self.mlp_allreduce_inputs,
            comm_overlap_window=self.overlap_window,
            comm_sm_configs=self.sm_configs,
        )
        return mlp_out

    def profile(self, monitor=None):
        if self.rank == 0:
            logs_dir = self._logs_dir()
            csv_path = os.path.join(logs_dir, "mlp_energy.csv")
            with open(csv_path, "w") as f:
                title = "time (s),total_energy (J)," + ",".join(
                    [f"rank{i} energy (J)" for i in range(self.world_size)]
                )
                f.write(title + "\n")

        # Warmup and rough iteration estimate
        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        dist.barrier()
        for i in range(10):
            if i == 2:
                torch.cuda.current_stream().synchronize()
                time_start = time.time()
            self._run_one_step()
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8.0
        torch.cuda.profiler.stop()

        if self.rank == 0:
            iterations = max(1, int(5.0 / max(duration, 1e-6)))
            dist_list = [iterations]
        else:
            dist_list = [None]
        dist.broadcast_object_list(dist_list, src=0, group=self.tp_group)
        iterations = dist_list[0]
        if self.rank == 0:
            print(f"[MLP] Per-step duration ~ {duration*1000:.3f} ms -> {iterations} iterations")

        torch.cuda.synchronize()
        dist.barrier()
        if self.rank == 0:
            monitor.begin_window("step")

        for _ in range(iterations):
            self._run_one_step()

        torch.cuda.synchronize()
        dist.barrier()

        if self.rank == 0:
            result = monitor.end_window("step")
            avg_time = result.time / iterations
            avg_total_energy = result.total_energy / iterations
            per_rank = [result.gpu_energy[i] / iterations for i in range(self.world_size)]

            logs_dir = self._logs_dir()
            csv_path = os.path.join(logs_dir, "mlp_energy.csv")
            with open(csv_path, "a") as f:
                f.write(
                    f"{avg_time},{avg_total_energy}," + ",".join(map(str, per_rank)) + "\n"
                )

            print(f"[MLP] avg_time={avg_time:.6f}s, avg_total_energy={avg_total_energy:.4f}J")


def _freq_sweep_worker(rank: int, world_size: int, args: argparse.Namespace, master_port: int, freq_values):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    try:
        tp_group = init_distributed(rank, world_size)
        profiler = MLPFuserProfiler(args, rank, world_size)
        monitor = ZeusMonitor(gpu_indices=list(range(world_size))) if rank == 0 else None
        for freq_mhz in freq_values:
            if rank == 0:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if visible is not None and len(visible.strip()) > 0:
                    vis_list = [int(x) for x in visible.split(",") if x.strip() != ""]
                    target_indices = vis_list
                else:
                    target_indices = None
                print(f"[MLPFuser] profiling at frequency {freq_mhz} MHz")
                _set_gpu_frequency(freq_mhz, device_indices=target_indices)
            dist.barrier(group=tp_group)

            profiler.frequency = freq_mhz
            profiler.profile(monitor)

            dist.barrier(group=tp_group)
            if rank == 0:
                time.sleep(5.0)
    except Exception as e:
        print(f"[rank {rank}] Error: {e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            pid = os.getpid()
            print(f"Killing process group {pid}")
            os.system(f"pkill -P {pid}")
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[rank {rank}] destroyed process group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--frequency", "-f", type=str, default="default", help="Initial label; overridden by sweep")

    args = parser.parse_args()
    print("Profiling MLP Fuser over frequency sweep")
    print(f"Model name: {args.model_name}")
    print(f"World size: {args.world_size}")
    print(f"TP size: {args.tensor_parallel_size}, CP size: {args.context_parallel_size}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Frequencies: {FREQ_VALUES}")

    from torch.multiprocessing import spawn
    spawn(
        _freq_sweep_worker,
        args=(
            args.world_size,
            args,
            random.randint(9000, 65000),
            FREQ_VALUES,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()


