import os
import sys
import time
import random
import traceback

import torch
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from zeus.monitor import ZeusMonitor
from cfuser.core.utils import nvtx_range


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


class PostprocessBackwardProfiler:
    def __init__(self, args, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device('cuda', rank)
        self.dtype = torch.bfloat16

        # Model dims
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = 3072
        self.num_attention_heads = 24
        self.vocab_size = args.vocab_size
        self.ffn_hidden_size = 8192

        # Config
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            ffn_hidden_size=self.ffn_hidden_size,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            rotary_interleaved=False,
            apply_rope_fusion=True,
            params_dtype=self.dtype,
            tensor_model_parallel_size=world_size,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )

        self.frequency = args.frequency

        # Op: output projection
        self.output_layer = ColumnParallelLinear(
            self.config.hidden_size,
            self.vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
        ).to(self.device)

        # Inputs and grad output
        torch.manual_seed(1234)
        self.hidden_states = torch.randn(
            self.seq_length,
            self.batch_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        self.tp_group = get_tensor_model_parallel_group()

        # Build graph once and cache outputs + grad tensor
        with nvtx_range('postprocess_forward_for_backward'):
            self.cached_logits, _ = self.output_layer(self.hidden_states, runtime_gather_output=None)
        self.cached_logits_grad = torch.randn_like(self.cached_logits, dtype=self.cached_logits.dtype)

    def _backward_step(self):
        with nvtx_range('postprocess_backward'):
            torch.autograd.backward(
                tensors=[self.cached_logits],
                grad_tensors=[self.cached_logits_grad],
                retain_graph=True,
            )

    def run(self):
        # Logs dir
        if self.rank == 0:
            os.makedirs(
                f"logs/tp{self.world_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}",
                exist_ok=True,
            )
            with open(
                f"logs/tp{self.world_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/postprocess_backward_energy.csv",
                'w',
            ) as f:
                title = "time (s),total_energy (J)," + ",".join(
                    [f"rank{i} energy (J)" for i in range(self.world_size)]
                )
                f.write(title + "\n")

        # Warmup + iteration estimate
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)
        for i in range(10):
            if i == 2:
                t0 = time.time()
            self._backward_step()
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)
        t1 = time.time()
        duration = (t1 - t0) / 8.0
        if self.rank == 0:
            iterations = max(1, int(8.0 / duration))
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
            self._backward_step()
        torch.cuda.synchronize()
        dist.barrier(group=self.tp_group)

        if self.rank == 0:
            result = monitor.end_window('step')
            t_result = result.time / iterations
            e_total = result.total_energy / iterations
            ranks_energy = [result.gpu_energy[i] / iterations for i in range(self.world_size)]
            with open(
                f"logs/tp{self.world_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/postprocess_backward_energy.csv",
                'a',
            ) as f:
                f.write(
                    f"{t_result},{e_total}," + ",".join(map(str, ranks_energy)) + "\n"
                )


def _worker(rank: int, world_size: int, args, master_port: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    try:
        init_distributed(rank, world_size)
        profiler = PostprocessBackwardProfiler(args, rank, world_size)
        profiler.run()
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


if __name__ == '__main__':
    import argparse
    from torch.multiprocessing import spawn

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', '-w', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--seq_len', '-s', type=int, default=4096)
    parser.add_argument('--vocab_size', '-v', type=int, default=128256)
    parser.add_argument('--frequency', '-f', type=str, default='default')
    args = parser.parse_args()

    print('Running postprocess backward profiling (output projection)')
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Frequency: {args.frequency}")

    spawn(
        _worker,
        args=(args.world_size, args, random.randint(8000, 65535)),
        nprocs=args.world_size,
        join=True,
    )


