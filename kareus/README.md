# kareus/ — Kareus runtime

This package implements Kareus on top of the vendored Megatron-LM / NeMo /
TransformerEngine stack in [3rdparty/](../3rdparty/). It is installed in
editable mode by the Docker build (see [pyproject.toml](../pyproject.toml)).

## Subdirectories

| Directory             | Role                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------- |
| `megatron/`           | Kareus extensions to megatron-core (transformer layers, parallel state, partitions).    |
| `megatron/core/partitions/` | **Core partition-overlap execution engine** (see below).                          |
| `scheduler/`          | Pipeline-comm scheduler that consumes `scheds_pipeline_*.py` solutions from the Phillips-Dessouky solver and drives the per-microbatch comm/compute interleaving. |
| `nemo/`               | NeMo glue: training-step hooks, Lightning-loop integration, kareus_scheduler kwargs.    |
| `transformer_engine/` | Adapter shims around the patched TransformerEngine modules (see [3rdparty/README.md](../3rdparty/README.md)). |
| `apex/`               | Adaptations of Apex transformer utilities used by megatron-core.                        |
| `flash_attn/`         | Adaptations around FlashAttention-2/3 (`hopper/`) interfaces.                           |
| `msccl/`              | Thin wrapper around `mscclpp` for partition-aware collective communication.             |
| `utils/`              | Debugging / tracing helpers (`debug.py`).                                               |

Most files in the per-layer adapter directories are **adaptations**, not new
algorithms — Kareus has to insert partition boundaries and frequency switches
at every layer of the stack, so each integration point has its own thin
shim.

## Core partition-overlap execution engine

The actual Kareus partition algorithm lives in
[`kareus/megatron/core/partitions/`](megatron/core/partitions/):

| File                      | What it does                                                              |
| ------------------------- | ------------------------------------------------------------------------- |
| `partition_base.py`       | Base class for a partition (one fwd or bwd subgraph that can run at a chosen frequency). |
| `partition_builder.py`    | Splits a transformer block / pipeline stage into the partition set used by the BO profilers and the scheduler. |
| `forward_partition.py`    | Forward-side partitions (`fwd_qkv_*`, `fwd_attn`, `fwd_ao_*`, `fwd_mlp`). |
| `backward_partition.py`   | Backward-side partitions (`bwd_qkv_*`, `bwd_o_*`, `bwd_a_*`, `bwd_mlp`).  |
| `context_manager.py`      | Per-partition context manager (sets GPU clock, swaps streams, fences communication). |
| `tensor_graph.py`         | Lightweight tensor-DAG used to wire partitions to the autograd graph.     |
| `autograd_function.py`    | Custom `torch.autograd.Function` that bridges a partition to PyTorch autograd. |

The same partition naming used here (`fwd_qkv_ar`, `bwd_a_rs`, ...) shows up
in the BO profilers under [tests/bayesian/partitions/](../tests/bayesian/partitions/)
and in the per-instruction frequency / schedule files under
[tests/kareus/schedules/](../tests/kareus/schedules/).
