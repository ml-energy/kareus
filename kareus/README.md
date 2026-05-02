# kareus/ — Kareus runtime

This package implements Kareus on top of the vendored Megatron-LM / NeMo /
TransformerEngine stack in [3rdparty/](../3rdparty/). 

## Subdirectories

| Directory             | Role                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------- |
| `megatron/`           | Kareus extensions to megatron-core (transformer layers, parallel state, partitions).    |
| `megatron/core/partitions/` | **Core partition-overlap execution engine** (see below).                          |
| `scheduler/`          | Pipeline-comm scheduler that consumes `scheds_pipeline_*.py` solutions from the Phillips-Dessouky solver and drives the per-microbatch comm/compute overlap. |
| `nemo/`               | NeMo glue: training-step hooks, Lightning-loop integration, kareus_scheduler kwargs.    |
| `transformer_engine/` | Adapter shims around the patched TransformerEngine modules.                             |
| `apex/`               | Adaptations of Apex transformer utilities used by megatron-core.                        |
| `flash_attn/`         | Adaptations around FlashAttention-2/3             interfaces.                           |
| `msccl/`              | Thin wrapper around `mscclpp` for partition-aware collective communication.             |
| `utils/`              | Debugging / tracing helpers.                                                            |

## Core partition-overlap execution engine

The actual Kareus partition algorithm lives in
[`kareus/megatron/core/partitions/`](megatron/core/partitions/):

At a high level, this package runs a transformer block as scheduled forward and
backward partitions over two nanobatches, enabling communication from one
nanobatch to overlap with compute from the other.

| File                      | Role                                                                      |
| ------------------------- | ------------------------------------------------------------------------- |
| `tensor_graph.py`         | Defines the graph representation used to describe partitionable compute and communication. |
| `partition_builder.py`    | Builds the forward and backward partition sequences consumed by the runtime scheduler. |
| `partition_base.py`       | Holds the common partition state and scheduling contract shared by forward and backward execution. |
| `forward_partition.py`    | Runs forward partition work and participates in scheduled communication overlap. |
| `backward_partition.py`   | Runs backward partition work, including gradient production, under the same overlap model. |
| `context_manager.py`      | Maintains per-nanobatch runtime state and saved activation state. |
| `autograd_function.py`    | Wraps the partitioned transformer block in one PyTorch autograd boundary. |
| `__init__.py`             | Exports the partition engine's public API. |
