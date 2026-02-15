# Step 7: TransformerBlock & TransformerLayer Integration

## Objective

Rewire `TransformerBlock` and `TransformerLayer` to use the new graph-based partition system. This is the final integration step that ties everything together.

## Dependencies

- ALL previous steps (1-6)

## Files to Modify

- `kareus/megatron/core/transformer/transformer_block.py` (major rewrite)
- `kareus/megatron/core/transformer/transformer_layer.py` (add `get_all_operators()`, remove fuser code)

---

## TransformerLayer Changes

### Add: `get_all_operators()`

New method that returns all `PartitionableOperator` instances in forward execution order.

```python
def get_all_operators(self) -> List[PartitionableOperator]:
    """Return all operators in forward execution order for this layer.

    Order for a standard LLaMA layer:
    1. attn_bda (BiasDropoutAddOp) — first layer uses identity
    2. attn_residual_fork (ResidualForkOp)
    3. input_layernorm (TENormOp)
    4. linear_qkv (TEColumnParallelLinearOp)
    5. qkv_postprocess (QKVPostProcessOp)
    6. rotary_embedding (RotaryEmbeddingOp)
    7. core_attention (TEDotProductAttentionOp)
    8. linear_proj (TERowParallelLinearOp)
    9. mlp_bda (BiasDropoutAddOp)
    10. mlp_residual_fork (ResidualForkOp)
    11. pre_mlp_layernorm (TENormOp)
    12. linear_fc1 (TEColumnParallelLinearOp)
    13. activation (BiasSwigluOp / BiasGeluOp / BiasGegluOp)
    14. linear_fc2 (TERowParallelLinearOp)
    """
    ops = []
    # Attention block
    ops.append(self.attn_bda)
    ops.append(self.attn_residual_fork)
    ops.append(self.input_layernorm)
    ops.append(self.linear_qkv)
    ops.append(self.qkv_postprocess)
    ops.append(self.rotary_embedding)
    ops.append(self.core_attention)
    ops.append(self.linear_proj)
    # MLP block
    ops.append(self.mlp_bda)
    ops.append(self.mlp_residual_fork)
    ops.append(self.pre_mlp_layernorm)
    ops.append(self.linear_fc1)
    ops.append(self.activation)
    ops.append(self.linear_fc2)
    return ops
```

### Remove (eventually)

- `build_fusers()`, `_build_attention_fusers()`, `_build_mlp_fusers()`
- `forward_attention()`, `forward_mlp()` and their TP/CP variants
- `init_tensor_parallel_comm()`, `init_context_parallel_comm()`
- `self.attention_fusers`, `self.mlp_fusers`

These are replaced by the block-level partition system. Keep them during migration for A/B testing.

---

## TransformerBlock Changes

### `__init__()` Flow

```python
def __init__(self, config, spec, post_layer_norm, pre_process, post_process):
    super().__init__(config=config)
    # ... (unchanged: submodules, post_layer_norm, pre_process, post_process)
    # ... (unchanged: checkpoint_core_attention, CPU offloading)

    # Build layers (unchanged)
    self._build_layers()

    # Scheduler (from config)
    self.scheduler = self.config.kareus_scheduler

    # NEW: Build tensor graphs and partitions
    self._build_partitions()
```

### New: `_build_partitions()`

```python
def _build_partitions(self):
    """Build TensorGraph and form partitions.

    Called from __init__ after layers are built.
    CommunicationOps have operator=None at this point.
    """
    # Step 1: Collect all operators, assign op_id
    all_ops = []
    for layer in self.layers:
        all_ops.extend(layer.get_all_operators())
    for idx, op in enumerate(all_ops):
        op.op_id = idx

    # Step 2a: Build FORWARD TensorGraph
    fwd_builder = TensorGraphBuilder()
    fwd_builder.add_initial_channels({
        "main": "t_input_0",
        "rotary_pos_emb": "t_rotary_pos_emb",
    })
    for op in all_ops:
        for fwd_spec in op.get_forward_ops():
            fwd_builder.add_op(fwd_spec)
    self.forward_tensor_graph = fwd_builder.build()

    # Step 2b: Build BACKWARD TensorGraph
    bwd_builder = TensorGraphBuilder()
    bwd_builder.add_initial_channels({
        "grad_main": "t_grad_output_0",
    })
    for op in reversed(all_ops):
        for bwd_spec in op.get_backward_ops():
            bwd_builder.add_op(bwd_spec)
    self.backward_tensor_graph = bwd_builder.build()

    # Step 3: Form partitions
    builder = PartitionBuilder(
        forward_tensor_graph=self.forward_tensor_graph,
        backward_tensor_graph=self.backward_tensor_graph,
        config=self.config,
    )
    self.forward_partitions = builder.build_forward_partitions()
    self.backward_partitions = builder.build_backward_partitions()
```

### Modified: `set_tensor_parallel_group()`

```python
def set_tensor_parallel_group(self, tp_group=None):
    """Create AllReduce operators and assign to partitions."""
    # Create 2 AllReduce ops (one per nanobatch) — same as current
    self.allreduce_comm_ops = [AllReduce(..., batch_idx=i) for i in range(2)]

    # NEW: Assign to CommunicationOps in partitions
    self._assign_comm_operators(CommunicationType.ALL_REDUCE, self.allreduce_comm_ops)
```

### Modified: `set_context_parallel_group()`

```python
def set_context_parallel_group(self, cp_group, cp_global_ranks, cp_stream):
    """Create AllGather/ReduceScatter operators and assign to partitions."""
    # Create 2 AllGatherKV + 2 ReduceScatterKV ops — same as current
    self.allgather_comm_ops = [AllGatherKV(..., batch_idx=i) for i in range(2)]
    self.reducescatter_comm_ops = [ReduceScatterKV(..., batch_idx=i) for i in range(2)]

    # NEW: Assign to CommunicationOps in partitions
    self._assign_comm_operators(CommunicationType.ALL_GATHER_KV, self.allgather_comm_ops)
    self._assign_comm_operators(CommunicationType.REDUCE_SCATTER_KV, self.reducescatter_comm_ops)
```

### New: `_assign_comm_operators()`

```python
def _assign_comm_operators(self, comm_type, comm_ops):
    """Assign physical comm operators to CommunicationOps of given type."""
    for partition in self.forward_partitions + self.backward_partitions:
        if partition.comm_op is not None and partition.comm_op.comm_type == comm_type:
            partition.comm_op.operator = comm_ops[partition.nano_batch_idx]
```

### Modified: `forward()`

```python
def forward(self, hidden_states, attention_mask, ..., rotary_pos_emb=None, ...):
    """Same external interface, new internal execution."""
    # ... (unchanged: inference_context, input_tensor, make_viewless_tensor)

    # Split into nano-batches
    (h1, *_), (h2, *_) = self._split_tensors_for_nanobatch(...)

    # Collect all params for autograd tracking
    all_params = list(self._get_all_params())

    # Execute through single autograd boundary
    h1_out, h2_out = TransformerBlockAutogradFunction.apply(
        h1, h2,
        rotary_pos_emb,
        attention_mask,
        self.forward_partitions,
        self.backward_partitions,
        self.forward_tensor_graph,
        self.backward_tensor_graph,
        self.scheduler,
        self.config,
        *all_params,
    )

    # Concatenate and apply final layernorm
    hidden_states = torch.cat([h1_out, h2_out], dim=1)
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, ...)

    return hidden_states
```

### New: `_get_all_params()`

```python
def _get_all_params(self) -> List[torch.nn.Parameter]:
    """Get all params in deterministic order for autograd tracking."""
    params = []
    for layer in self.layers:
        params.extend(layer.parameters())
    return params
```

### Remove (eventually)

- `_checkpointed_forward()`
- `_init_layer_tensor_parallel_comm()` → replaced by `_assign_comm_operators`
- `_init_context_parallel_comm()` → replaced by `_assign_comm_operators`
- The old forward loop that manually calls `layer.forward_attention()` / `layer.forward_mlp()`

---

## Initialization Flow Summary

```
1. TransformerBlock.__init__()
   a. _build_layers()                         → create TransformerLayer instances
   b. _build_partitions()                     → build graphs, form partitions
      (CommunicationOps have operator=None)

2. set_tensor_parallel_group()                → create AllReduce ops, assign to partitions
3. set_context_parallel_group()               → create AllGather/RS ops, assign to partitions

4. forward()                                  → split NB → autograd function → concat → layernorm
```

## Verification Criteria

- `TransformerBlock.__init__()` succeeds (graphs built, partitions formed)
- `set_tensor_parallel_group()` / `set_context_parallel_group()` correctly assigns comm ops
- `forward()` produces correct output matching the old implementation
- Backward produces correct gradients matching the old implementation
- End-to-end test: `tests/llama/kareus_gpt_pretraining.py` produces matching loss curves

## Migration Strategy

1. Keep old code alongside new code behind a flag (e.g., `config.use_graph_partitions`)
2. Run A/B comparison tests:
   - Same input → same output (forward)
   - Same grad_output → same grad_input and grad_params (backward)
3. Once verified, remove old fuser code

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 49-581 (Objective 1).
