import os
import sys
import time
import random
import traceback

import torch
import torch.distributed as dist
import multiprocessing as mp
import pynvml

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from megatron.core.transformer.transformer_config import TransformerConfig
from common_config import FuserTestConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.extensions.transformer_engine import TENorm
from zeus.monitor import ZeusMonitor
# from cfuser.core.utils import nvtx_range


# Frequency sweep configuration (A100-style: 1410, 1380, ..., 930 MHz)
FREQ_VALUES = list(range(1410, 890, -30))


def _set_gpu_frequency(target_freq_mhz, device_indices=None):
    """Best-effort GPU application clock setter via NVML."""
    pynvml.nvmlInit()
    all_indices = list(range(pynvml.nvmlDeviceGetCount()))
    indices = device_indices if device_indices is not None else all_indices
    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(target_freq_mhz), int(target_freq_mhz))
        time.sleep(1)
    pynvml.nvmlShutdown()


def _kill_all_subprocesses(timeout: float = 2.0):
    """Best-effort termination of any child processes spawned by this process.

    Attempts both multiprocessing-aware termination and psutil-based recursive kill.
    Safe to call from workers and from the main process.
    """
    try:
        for child in mp.active_children():
            try:
                child.terminate()
            except Exception:
                pass
        for child in mp.active_children():
            try:
                child.join(timeout)
            except Exception:
                pass

        try:
            import psutil  # type: ignore
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for proc in children:
                try:
                    proc.terminate()
                except Exception:
                    pass
            _, alive = psutil.wait_procs(children, timeout=timeout)
            for proc in alive:
                try:
                    proc.kill()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


def init_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(random.randint(8000, 65535)))

    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        create_gloo_process_groups=False,
    )


class PostprocessProfiler:
    def __init__(self, args, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.context_parallel_size = args.context_parallel_size
        self.tensor_parallel_size = args.tensor_parallel_size
        self.device = torch.device('cuda', rank)
        self.dtype = torch.bfloat16
        self.model_name = args.model_name
        
        # Model dims
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.vocab_size = args.vocab_size
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE

        # Config
        self.config = FuserTestConfig.create_postprocess_config(
            context_parallel_size=self.context_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
        )

        # Force RMSNorm for this profiler
        self.config.normalization = "RMSNorm"

        self.frequency = args.frequency

        # Ops: mimic GPTModel.output_layer
        self.output_layer = ColumnParallelLinear(
            self.config.hidden_size,
            self.vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,  # parallel_output=True behavior
        )

        self.output_layer.to(self.device)

        # Inputs: hidden states [s, b, h]
        torch.manual_seed(1234)
        local_seq_length = self.seq_length // max(1, self.context_parallel_size)
        nano_batches = 2
        nb_batch_size = self.batch_size // nano_batches
        self.hidden_states_nb0 = torch.randn(
            local_seq_length,
            nb_batch_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.hidden_states_nb1 = torch.randn(
            local_seq_length,
            nb_batch_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.rms_eps = 1e-5

        # TENorm (RMSNorm)
        self.norm = TENorm(self.config, hidden_size=self.hidden_size, eps=self.rms_eps)
        self.norm.to(self.device)

        # Parallel group
        self.tp_group = get_tensor_model_parallel_group()

    def _forward_step(self):
        x = torch.cat((self.hidden_states_nb0, self.hidden_states_nb1), dim=1)
        x = self.norm(x)
        logits, _ = self.output_layer(x, runtime_gather_output=None)
        return logits

    def run(self):
        # Logs dir
        if self.rank == 0:
            os.makedirs(
                f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}",
                exist_ok=True,
            )
            with open(
                f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/postprocess_energy.csv",
                'w',
            ) as f:
                title = "time (s),total_energy (J)," + ",".join(
                    [f"rank{i} energy (J)" for i in range(self.world_size)]
                )
                f.write(title + "\n")

        # Warmup and iterations estimate
        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)
        for i in range(10):
            if i == 2:
                t0 = time.time()
            _ = self._forward_step()
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)
        t1 = time.time()
        duration = (t1 - t0) / 8.0
        torch.cuda.profiler.stop()
        if self.rank == 0:
            iterations = max(1, int(5.0 / duration))
            dist_list = [iterations]
        else:
            dist_list = [None]
        dist.broadcast_object_list(dist_list, src=0, group=self.tp_group)
        iterations = dist_list[0]

        monitor = None
        if self.rank == 0:
            monitor = ZeusMonitor(gpu_indices=list(range(self.world_size)))

        # Measure
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)
        if self.rank == 0:
            monitor.begin_window('step')
        for _ in range(iterations):
            _ = self._forward_step()
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)

        if self.rank == 0:
            result = monitor.end_window('step')
            t_result = result.time / iterations
            e_total = result.total_energy / iterations
            ranks_energy = [result.gpu_energy[i] / iterations for i in range(self.world_size)]
            with open(
                f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/postprocess_energy.csv",
                'a',
            ) as f:
                f.write(
                    f"{t_result},{e_total}," + ",".join(map(str, ranks_energy)) + "\n"
                )


def _freq_sweep_worker(rank: int, world_size: int, args, master_port: int, freq_values):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    try:
        init_distributed(rank, world_size)
        # Sweep all configured frequencies in a single distributed run
        for freq_mhz in freq_values:
            if rank == 0:
                # Set GPU frequency via NVML (all visible devices or CUDA_VISIBLE_DEVICES)
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                if visible is not None and len(visible.strip()) > 0:
                    vis_list = [int(x) for x in visible.split(',') if x.strip() != '']
                    target_indices = vis_list
                else:
                    target_indices = None
                print(f'PostprocessProfiler: profiling at frequency {freq_mhz} MHz')
                _set_gpu_frequency(freq_mhz, device_indices=target_indices)
            # Ensure all ranks are synchronized before profiling this frequency
            dist.barrier()

            # Update args.frequency for logging paths and construct profiler
            args.frequency = str(freq_mhz)
            profiler = PostprocessProfiler(args, rank, world_size)
            profiler.run()

            # Small pause between frequency changes
            dist.barrier()
            if rank == 0:
                time.sleep(5.0)
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        traceback.print_exc()
    finally:
        try:
            if dist.is_initialized():
                destroy_model_parallel()
                dist.destroy_process_group()
        except Exception:
            pass
        _kill_all_subprocesses()


if __name__ == '__main__':
    import argparse
    from torch.multiprocessing import spawn

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument('--world_size', '-w', type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument('--context_parallel_size', '-c', type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument('--tensor_parallel_size', '-t', type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument('--batch_size', '-b', type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument('--seq_len', '-s', type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument('--vocab_size', '-v', type=int, default=FuserTestConfig.VOCAB_SIZE)
    # Kept for compatibility but ignored during sweep; per-run frequency is set internally.
    parser.add_argument('--frequency', '-f', type=str, default='default')
    args = parser.parse_args()

    print('Running postprocess profiling (output projection) over frequency sweep')
    print(f"Model name: {args.model_name}")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Frequencies: {FREQ_VALUES}")

    spawn(
        _freq_sweep_worker,
        args=(args.world_size, args, random.randint(8000, 65535), FREQ_VALUES),
        nprocs=args.world_size,
        join=True,
    )
    _kill_all_subprocesses()


