# Step 6: Unified Autograd Function — `autograd_function.py`

## Objective

Create `TransformerBlockAutogradFunction` — a single `torch.autograd.Function` that wraps the ENTIRE transformer block execution (all layers, all partitions, both nanobatches).

## Dependencies

- Step 1: `TensorGraph`
- Step 2: `NanoBatchContext`
- Step 4: `ForwardPartition`, `BackwardPartition`
- Step 5: `PartitionBuilder` (partitions are already formed when this runs)

## File to Create

`kareus/megatron/core/partitions/autograd_function.py`

## Why a Single Autograd Function?

**Current architecture** has separate autograd functions per fuser type:
- `_PartitionFuserAutogradFunction`
- `_QKVFuserAutogradFunction`
- `_AttnOprojFuserAutogradFunction`

Each manages its own `save_for_backward` and gradient computation. This is fragile:
- Cross-partition tensor flow (bias, residual) requires manual plumbing between fusers
- Parameter gradients must be manually accumulated across fusers
- TransformerBlock forward loop must manually manage residual/comm state variables

**New design:** One autograd boundary wrapping everything → all tensor routing is automatic via TensorStore + tensor_ids.

---

## `TransformerBlockAutogradFunction`

```python
class TransformerBlockAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(func_ctx, h1, h2, rotary_pos_emb, attention_mask,
                forward_partitions, backward_partitions,
                forward_tensor_graph, backward_tensor_graph,
                scheduler, config, *params): ...

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(func_ctx, grad_h1, grad_h2): ...
```

---

### Forward

**Inputs:**
- `h1, h2`: Nano-batch input tensors `[s, b/2, h]`
- `rotary_pos_emb`: Shared across both NB
- `attention_mask`: Shared
- `forward_partitions`: List of `ForwardPartition` (already interleaved NB1/NB2)
- `backward_partitions`: Saved for backward
- `forward_tensor_graph`, `backward_tensor_graph`: For final tensor_id lookup
- `scheduler`: `PipelineCommScheduler` with `current_schedule`
- `config`: `TransformerConfig`
- `*params`: All parameters from all layers (for autograd tracking)

**Execution steps:**

```
1. Create NanoBatchContexts:
   ctx_nb1 = NanoBatchContext(batch_idx=0)
   ctx_nb2 = NanoBatchContext(batch_idx=1)

2. Seed TensorStores with initial tensors:
   ctx_nb1.tensor_store.set("t_input_0", h1)
   ctx_nb2.tensor_store.set("t_input_0", h2)
   ctx_nb1.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)
   ctx_nb2.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)

3. Load schedule from scheduler.current_schedule

4. Execute forward partitions in order:
   for partition in forward_partitions:
       partition.load_schedule(current_schedule)
       if partition.nano_batch_idx == 0:
           partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
       else:
           partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)

5. Extract final outputs:
   final_tid = forward_tensor_graph.get_output_channel("main")
   h1_out = ctx_nb1.tensor_store.get(final_tid)
   h2_out = ctx_nb2.tensor_store.get(final_tid)

6. Save for backward:
   func_ctx.backward_partitions = backward_partitions
   func_ctx.backward_tensor_graph = backward_tensor_graph
   func_ctx.scheduler = scheduler
   func_ctx.nano_ctx_1 = ctx_nb1
   func_ctx.nano_ctx_2 = ctx_nb2
   func_ctx.num_params = len(params)
   saved_1 = ctx_nb1.flatten_saved_tensors()
   saved_2 = ctx_nb2.flatten_saved_tensors()
   func_ctx.num_saved_1 = len(saved_1)
   func_ctx.save_for_backward(*saved_1, *saved_2)

7. Return (h1_out, h2_out)
```

---

### Backward

**Inputs:**
- `grad_h1, grad_h2`: Gradients of loss w.r.t. forward outputs

**Execution steps:**

```
1. Restore saved tensors:
   saved = func_ctx.saved_tensors
   ctx_nb1.restore_saved_tensors(saved[:num_saved_1])
   ctx_nb2.restore_saved_tensors(saved[num_saved_1:])

2. Seed backward TensorStores:
   ctx_nb1.tensor_store.set("t_grad_output_0", grad_h1)
   ctx_nb2.tensor_store.set("t_grad_output_0", grad_h2)

3. Load schedule

4. Execute backward partitions:
   all_grad_params_nb1 = {}
   all_grad_params_nb2 = {}
   for partition in backward_partitions:
       partition.load_schedule(current_schedule)
       if partition.nano_batch_idx == 0:
           grad_params = partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
       else:
           grad_params = partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
       # Accumulate grad_params by nanobatch
       target = all_grad_params_nb1 if nb==0 else all_grad_params_nb2
       target.update(grad_params)

5. Extract final input gradients:
   final_grad_tid = backward_tensor_graph.get_output_channel("grad_main")
   dh1 = ctx_nb1.tensor_store.get(final_grad_tid)
   dh2 = ctx_nb2.tensor_store.get(final_grad_tid)

6. Merge parameter gradients:
   combined = _combine_param_grads(nb1_grads, nb2_grads, num_params)

7. Return (dh1, dh2, None, None, ..., *combined)
   # None for non-differentiable inputs (rotary, mask, partitions, graphs, etc.)
```

---

## `_combine_param_grads()`

Helper function to merge parameter gradients from both nanobatches.

```python
def _combine_param_grads(
    grad_params_nb1: Dict[int, List[Optional[Tensor]]],
    grad_params_nb2: Dict[int, List[Optional[Tensor]]],
    num_params: int,
) -> List[Optional[Tensor]]:
```

**Logic:**
1. Build flat list of size `num_params` (initialized to None)
2. Iterate sorted op_ids
3. For each op_id: sum g1 + g2 for corresponding parameters (in-place add)
4. Return flat list matching the `*params` order from forward

**Ordering guarantee:** op_ids are assigned sequentially by `_build_partitions` in layer→operator order, matching `layer.parameters()` iteration order used in `_get_all_params()`.

---

## Return Value Structure

The backward return must match the forward signature:
```
(h1, h2, rotary_pos_emb, attention_mask,
 forward_partitions, backward_partitions,
 forward_tensor_graph, backward_tensor_graph,
 scheduler, config, *params)
```

Returns:
```
(dh1, dh2, None, None, None, None, None, None, None, None, *combined_grad_params)
```

## Verification Criteria

- Forward populates both NanoBatchContexts correctly
- `save_for_backward` captures all needed tensors
- Backward restores contexts and produces correct gradients
- `_combine_param_grads` correctly sums NB1 + NB2 gradients
- Return tuple matches forward signature exactly
- Gradient flow verified end-to-end with `torch.autograd.gradcheck`

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 2135-2519 (Objective 5).
