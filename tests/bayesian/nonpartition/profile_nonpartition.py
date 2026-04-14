import os
import sys
import time
import random
import traceback

import torch
import torch.distributed as dist
import multiprocessing as mp
from torch.multiprocessing import spawn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Add tests/bayesian/ so the "common" package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from megatron.core.parallel_state import (
    initialize_model_parallel,
    destroy_model_parallel,
)
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.language_module.language_module import LanguageModule
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from zeus.monitor import ZeusMonitor
from zeus.profile import measure as zeus_measure

from common.model_config import get_model_config, GPU_CONFIGS
from common.hardware import _set_gpu_frequency, reset_gpu_clocks, get_visible_gpu_indices


def _kill_all_subprocesses(timeout: float = 2.0):
    """Best-effort termination of any child processes spawned by this process."""
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


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _log_path(args, freq, name):
    """Return (dir_path, csv_path) for a given profiler name."""
    d = (
        f"logs/{args.model_name}"
        f"/cp{args.context_parallel_size}-tp{args.tensor_parallel_size}"
        f"-bs{args.batch_size}-seq{args.seq_len}"
        f"/{freq}"
    )
    return d, os.path.join(d, f"{name}_energy.csv")


def _write_header(csv_path):
    with open(csv_path, 'w') as f:
        f.write("time (s),total_energy (J),per_rank_energy (J)\n")


def _write_row(csv_path, t_result, e_total, world_size):
    with open(csv_path, 'a') as f:
        f.write(f"{t_result},{e_total},{e_total / world_size}\n")


# ---------------------------------------------------------------------------
# Measurement helper — mirrors partition's hardware.py pattern
# ---------------------------------------------------------------------------

def _measure(forward_fn, rank, world_size, measurement_duration=5.0, cooldown_duration=5.0):
    """Measure energy/time via zeus_measure (one ZeusMonitor per rank).

    Mirrors the pattern in common/hardware.py:_dist_batch_eval_worker:
      - Each rank owns its own ZeusMonitor(gpu_indices=[rank])
      - All ranks call zeus_measure; trial.energy_per_iter is already the
        total energy across all ranks (zeus_measure all-reduces internally)
      - Only rank 0 reads and returns the result
    """
    monitor = ZeusMonitor(gpu_indices=[rank])
    dist.barrier()

    trial = zeus_measure(
        target_function=forward_fn,
        zeus_monitor=monitor,
        measurement_duration=measurement_duration,
        cooldown_duration=cooldown_duration,
    )

    del monitor

    if rank == 0:
        return float(trial.time_per_iter), float(trial.energy_per_iter)
    return None, None


# ---------------------------------------------------------------------------
# Profiler 1: Embedding forward
# ---------------------------------------------------------------------------

class PreprocessProfiler:
    def __init__(self, args, model_cfg, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.device = torch.device('cuda', rank)

        tp = args.tensor_parallel_size
        cp = args.context_parallel_size
        config = model_cfg.create_transformer_config(tp, cp, use_cpu_initialization=True)

        local_seq = args.seq_len // max(1, cp)
        self.embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=model_cfg.vocab_size,
            max_sequence_length=local_seq,
            position_embedding_type='rope',
            scatter_to_sequence_parallel=False,
        ).to(self.device)

        self.input_ids = torch.randint(
            0, model_cfg.vocab_size,
            (args.batch_size, local_seq),
            device=self.device, dtype=torch.long,
        )
        self.position_ids = torch.arange(local_seq, device=self.device, dtype=torch.long)[
            None, :
        ].expand(args.batch_size, -1)

        self._args = args

    def _step(self):
        return self.embedding(self.input_ids, self.position_ids)

    def run(self):
        args = self._args
        freq = args.frequency
        if self.rank == 0:
            d, p = _log_path(args, freq, 'preprocess')
            os.makedirs(d, exist_ok=True)
            _write_header(p)

        t, e = _measure(self._step, self.rank, self.world_size)
        if self.rank == 0:
            _, p = _log_path(args, freq, 'preprocess')
            _write_row(p, t, e, self.world_size)
            print(f"PreprocessProfiler: time={t:.6f}s energy={e:.3f}J")


# ---------------------------------------------------------------------------
# Profiler 2: Embedding backward
# ---------------------------------------------------------------------------

class PreprocessBackwardProfiler:
    def __init__(self, args, model_cfg, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.device = torch.device('cuda', rank)

        tp = args.tensor_parallel_size
        cp = args.context_parallel_size
        config = model_cfg.create_transformer_config(tp, cp, use_cpu_initialization=True)

        local_seq = args.seq_len // max(1, cp)
        self.embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=model_cfg.vocab_size,
            max_sequence_length=local_seq,
            position_embedding_type='rope',
            scatter_to_sequence_parallel=False,
        ).to(self.device)

        self.input_ids = torch.randint(
            0, model_cfg.vocab_size,
            (args.batch_size, local_seq),
            device=self.device, dtype=torch.long,
        )
        self.position_ids = torch.arange(local_seq, device=self.device, dtype=torch.long)[
            None, :
        ].expand(args.batch_size, -1)

        # Pre-run forward once; cache output and grad tensor for repeated backward
        self.cached_embeddings = self.embedding(self.input_ids, self.position_ids)
        self.cached_grad = torch.randn_like(self.cached_embeddings)

        self._args = args

    def _step(self):
        torch.autograd.backward(
            tensors=[self.cached_embeddings],
            grad_tensors=[self.cached_grad],
            retain_graph=True,
        )
        # torch.autograd.backward(
        #     tensors=[self.cached_embeddings],
        #     grad_tensors=[self.cached_grad],
        #     retain_graph=True,
        # )

    def run(self):
        args = self._args
        freq = args.frequency
        if self.rank == 0:
            d, p = _log_path(args, freq, 'preprocess_backward')
            os.makedirs(d, exist_ok=True)
            _write_header(p)

        t, e = _measure(self._step, self.rank, self.world_size)
        if self.rank == 0:
            _, p = _log_path(args, freq, 'preprocess_backward')
            _write_row(p, t, e, self.world_size)
            print(f"PreprocessBackwardProfiler: time={t:.6f}s energy={e:.3f}J")


# ---------------------------------------------------------------------------
# Profiler 3: Output layer forward (TENorm + ColumnParallelLinear)
# ---------------------------------------------------------------------------

class PostprocessProfiler:
    def __init__(self, args, model_cfg, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.device = torch.device('cuda', rank)
        self.dtype = torch.bfloat16

        tp = args.tensor_parallel_size
        cp = args.context_parallel_size
        config = model_cfg.create_transformer_config(
            tp, cp, use_cpu_initialization=True, normalization="RMSNorm"
        )

        hidden_size = model_cfg.hidden_size
        local_seq = args.seq_len // max(1, cp)
        nano_batches = 2
        nb_bs = args.batch_size // nano_batches

        self.norm = TENorm(config, hidden_size=hidden_size, eps=model_cfg.layernorm_epsilon).to(self.device)
        self.output_layer = ColumnParallelLinear(
            hidden_size,
            model_cfg.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
        ).to(self.device)

        self.hidden_nb0 = torch.randn(local_seq, nb_bs, hidden_size, dtype=self.dtype, device=self.device)
        self.hidden_nb1 = torch.randn(local_seq, nb_bs, hidden_size, dtype=self.dtype, device=self.device)

        self._args = args

    def _step(self):
        x = torch.cat((self.hidden_nb0, self.hidden_nb1), dim=1)
        x = self.norm(x)
        logits, _ = self.output_layer(x, runtime_gather_output=None)
        return logits

    def run(self):
        args = self._args
        freq = args.frequency
        if self.rank == 0:
            d, p = _log_path(args, freq, 'postprocess')
            os.makedirs(d, exist_ok=True)
            _write_header(p)

        t, e = _measure(self._step, self.rank, self.world_size)
        if self.rank == 0:
            _, p = _log_path(args, freq, 'postprocess')
            _write_row(p, t, e, self.world_size)
            print(f"PostprocessProfiler: time={t:.6f}s energy={e:.3f}J")


# ---------------------------------------------------------------------------
# Profiler 4: Output layer backward (kareus RMSNorm + ColumnParallelLinear)
# ---------------------------------------------------------------------------

class PostprocessBackwardProfiler:
    def __init__(self, args, model_cfg, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.device = torch.device('cuda', rank)
        self.dtype = torch.bfloat16

        tp = args.tensor_parallel_size
        cp = args.context_parallel_size
        config = model_cfg.create_transformer_config(
            tp, cp, use_cpu_initialization=True, normalization="RMSNorm"
        )

        hidden_size = model_cfg.hidden_size
        local_seq = args.seq_len // max(1, cp)
        nano_batches = 2
        nb_bs = args.batch_size // nano_batches

        self.norm = RMSNorm(
            normalized_shape=hidden_size,
            eps=model_cfg.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype,
        )
        self.output_layer = ColumnParallelLinear(
            hidden_size,
            model_cfg.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
        ).to(self.device)

        self.hidden_states = torch.randn(
            local_seq, args.batch_size, hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        self.hidden_nb0 = torch.randn(
            local_seq, nb_bs, hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )
        self.hidden_nb1 = torch.randn(
            local_seq, nb_bs, hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True,
        )

        # Pre-run forward; cache logits and grad tensor for repeated backward
        x = self.norm(self.hidden_states)
        self.cached_logits, _ = self.output_layer(x, runtime_gather_output=None)
        self.cached_logits_grad = torch.randn_like(self.cached_logits)

        self._args = args

    def _step(self):
        self.cached_logits_grad.mul_(0.5)
        _ = torch.autograd.grad(
            outputs=[self.cached_logits],
            inputs=[self.hidden_states],
            grad_outputs=[self.cached_logits_grad],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        # Dummy add to simulate nano-batch grad accumulation
        _ = self.hidden_nb0 + self.hidden_nb1

    def run(self):
        args = self._args
        freq = args.frequency
        if self.rank == 0:
            d, p = _log_path(args, freq, 'postprocess_backward')
            os.makedirs(d, exist_ok=True)
            _write_header(p)

        self.output_layer.zero_grad(set_to_none=True)
        t, e = _measure(self._step, self.rank, self.world_size)
        if self.rank == 0:
            _, p = _log_path(args, freq, 'postprocess_backward')
            _write_row(p, t, e, self.world_size)
            print(f"PostprocessBackwardProfiler: time={t:.6f}s energy={e:.3f}J")

        # Explicit GPU memory cleanup before this object is GC'd
        self.output_layer.zero_grad(set_to_none=True)
        self.cached_logits = self.cached_logits.detach()
        self.cached_logits_grad = None
        self.hidden_nb0 = None
        self.hidden_nb1 = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Profiler 5: Cross-entropy loss
# ---------------------------------------------------------------------------

class LossProfiler:
    def __init__(self, args, model_cfg, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.device = torch.device('cuda', rank)
        self.dtype = torch.bfloat16

        tp = args.tensor_parallel_size
        cp = args.context_parallel_size
        config = model_cfg.create_transformer_config(
            tp, cp,
            use_cpu_initialization=True,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl='te',
        )

        local_seq = args.seq_len // max(1, cp)
        vocab_per_partition = model_cfg.vocab_size // max(1, tp)

        self.loss_mod = LanguageModule(config).to(self.device)

        self.logits = torch.randn(
            local_seq, args.batch_size, vocab_per_partition,
            dtype=self.dtype, device=self.device,
        )
        self.labels = torch.randint(
            0, model_cfg.vocab_size,
            (args.batch_size, local_seq),
            device=self.device, dtype=torch.long,
        )

        self._args = args

    def _step(self):
        return self.loss_mod.compute_language_model_loss(self.labels, self.logits)

    def run(self):
        args = self._args
        freq = args.frequency
        if self.rank == 0:
            d, p = _log_path(args, freq, 'loss')
            os.makedirs(d, exist_ok=True)
            _write_header(p)

        t, e = _measure(self._step, self.rank, self.world_size)
        if self.rank == 0:
            _, p = _log_path(args, freq, 'loss')
            _write_row(p, t, e, self.world_size)
            print(f"LossProfiler: time={t:.6f}s energy={e:.3f}J")


# ---------------------------------------------------------------------------
# Frequency sweep worker
# ---------------------------------------------------------------------------

PROFILER_CLASSES = [
    PreprocessProfiler,
    PreprocessBackwardProfiler,
    PostprocessProfiler,
    PostprocessBackwardProfiler,
    LossProfiler,
]


def _freq_sweep_worker(rank: int, world_size: int, args, master_port: int, freq_values):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    try:
        init_distributed(rank, world_size)
        model_cfg = get_model_config(args.model_name)

        for freq_mhz in freq_values:
            if rank == 0:
                print(f'[nonpartition] Profiling at {freq_mhz} MHz')
                _set_gpu_frequency(freq_mhz, device_indices=get_visible_gpu_indices())
            dist.barrier()

            args.frequency = str(freq_mhz)
            for ProfilerCls in PROFILER_CLASSES:
                p = ProfilerCls(args, model_cfg, rank, world_size)
                p.run()

            dist.barrier()
            time.sleep(5.0)

    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            dist.destroy_process_group()
        _kill_all_subprocesses()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Nonpartition energy profiling (embedding, output layer, loss)')
    parser.add_argument('--model_name', '-m', type=str, required=True,
                        choices=list(__import__('common.model_config', fromlist=['MODEL_REGISTRY']).MODEL_REGISTRY))
    parser.add_argument('--world_size', '-w', type=int, required=True)
    parser.add_argument('--tensor_parallel_size', '-tp', type=int, required=True)
    parser.add_argument('--context_parallel_size', '-cp', type=int, required=True)
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--seq_len', '-s', type=int, required=True)
    parser.add_argument('--gpu_type', type=str, default='A100', choices=list(GPU_CONFIGS))
    args = parser.parse_args()

    freq_min, freq_max, freq_step = GPU_CONFIGS[args.gpu_type]["freq_range"]
    freq_values = list(range(freq_max, freq_min - freq_step, -freq_step))

    print(f'Running nonpartition profiling')
    print(f'  model:      {args.model_name}')
    print(f'  world_size: {args.world_size}  tp: {args.tensor_parallel_size}  cp: {args.context_parallel_size}')
    print(f'  batch_size: {args.batch_size}  seq_len: {args.seq_len}')
    print(f'  gpu_type:   {args.gpu_type}  frequencies: {freq_values}')

    spawn(
        _freq_sweep_worker,
        args=(args.world_size, args, random.randint(8000, 65535), freq_values),
        nprocs=args.world_size,
        join=True,
    )
    reset_gpu_clocks(device_indices=get_visible_gpu_indices())
    _kill_all_subprocesses()
