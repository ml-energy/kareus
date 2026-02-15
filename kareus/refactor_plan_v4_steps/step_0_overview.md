# Refactor Plan v4 - Overview & Dependency Map

## Goal

Replace the manual partition/fuser system with an automatic graph-based partition system. The external `TransformerBlock` interface stays the same; internals change from manual fuser composition to: **Build TensorGraph -> Form Partitions -> Execute with Overlap**.

## Implementation Order (Dependency-Driven)

```
Step 1: tensor_graph.py          (core data structures, no deps)
   |
Step 2: context_manager.py       (TensorStore, NanoBatchContext; depends on Step 1 types)
   |
Step 3: operator migration       (add PartitionableOperator to all ops; depends on Step 1)
   |
Step 4: partition classes         (ForwardPartition, BackwardPartition; depends on Steps 1-2)
   |
Step 5: partition_builder.py     (auto partition formation; depends on Steps 1, 4)
   |
Step 6: autograd_function.py     (unified autograd boundary; depends on Steps 1-5)
   |
Step 7: transformer integration  (TransformerBlock + TransformerLayer rewire; depends on all)
```

## Dependency Graph

```
Step 1 ──────┬──── Step 2 ──── Step 4 ──── Step 5
             │                    │            │
             └──── Step 3        │            │
                                  └────────────┼──── Step 6 ──── Step 7
```

## New Files to Create

| File | Step | Description |
|------|------|-------------|
| `kareus/megatron/core/partitions/__init__.py` | 1 | Package init |
| `kareus/megatron/core/partitions/tensor_graph.py` | 1 | TensorGraph, TensorGraphBuilder, Channel, etc. |
| `kareus/megatron/core/partitions/context_manager.py` | 2 | TensorStore, NanoBatchContext |
| `kareus/megatron/core/partitions/forward_partition.py` | 4 | ForwardPartition |
| `kareus/megatron/core/partitions/backward_partition.py` | 4 | BackwardPartition |
| `kareus/megatron/core/partitions/partition_builder.py` | 5 | PartitionBuilder |
| `kareus/megatron/core/partitions/autograd_function.py` | 6 | TransformerBlockAutogradFunction |
| `kareus/megatron/core/extensions/ops/residual_fork.py` | 3 | ResidualForkOp (already exists, needs update) |

## Existing Files to Modify

| File | Step | Changes |
|------|------|---------|
| `kareus/megatron/core/extensions/ops/te_linear.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/te_norm.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/te_attention.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/bias_dropout_add.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/qkv_postprocess.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/rotary_embedding.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/bias_swiglu.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/bias_gelu.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/bias_geglu.py` | 3 | Add PartitionableOperator mixin |
| `kareus/megatron/core/extensions/ops/__init__.py` | 3 | Export new types |
| `kareus/megatron/core/transformer/transformer_layer.py` | 7 | Add `get_all_operators()`, remove fuser building |
| `kareus/megatron/core/transformer/transformer_block.py` | 7 | Replace loop with graph-based execution |

## Files to Eventually Remove (after migration)

| File | Reason |
|------|--------|
| `kareus/megatron/core/extensions/fusers/partition_fuser.py` | Replaced by ForwardPartition + BackwardPartition |
| `kareus/megatron/core/extensions/fusers/qkv_fuser.py` | Replaced by automatic partition formation |
| `kareus/megatron/core/extensions/fusers/qkv_fuser2.py` | Replaced by automatic partition formation |
| `kareus/megatron/core/extensions/fusers/attn_oproj_fuser.py` | Replaced by automatic partition formation |
| `kareus/megatron/core/extensions/fusers/partition_fuser_profile.py` | Replaced by automatic partition formation |

## Testing Strategy

Each step should be testable incrementally:
- Steps 1-2: Unit test data structures (TensorGraphBuilder routing, TensorStore)
- Step 3: Unit test operator channel declarations
- Steps 4-5: Unit test partition formation from a small graph
- Step 6: Integration test with mock partitions
- Step 7: Full end-to-end test using `tests/llama/kareus_gpt_pretraining.py`
