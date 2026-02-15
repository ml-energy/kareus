# Step 4: Forward & Backward Partition Classes

## Objective

Create `ForwardPartition` and `BackwardPartition` — the execution units that run compute ops on one nanobatch while overlapping communication for the other nanobatch.

## Dependencies

- Step 1: Uses `ComputeOp`, `CommunicationOp`, `TensorPort`
- Step 2: Uses `NanoBatchContext`, `TensorStore`

## Files to Create

- `kareus/megatron/core/partitions/forward_partition.py`
- `kareus/megatron/core/partitions/backward_partition.py`

## Type Aliases

```python
OverlapWindow = Tuple[int, int]   # (comm_start, comm_end) — fused_idx to launch/finish comm
ResourceShape = Tuple[int, int]   # (sm_num, block_size) — SM allocation for comm
```

---

## `ForwardPartition`

```python
@dataclass
class ForwardPartition:
    partition_id: int
    partition_key: str           # For scheduler lookup
    nano_batch_idx: int          # 0 or 1
    comp_ops: List[ComputeOp]    # Compute ops for THIS nanobatch
    comm_op: Optional[CommunicationOp]  # Comm op for the OTHER nanobatch
    _schedule_config: Optional[Tuple[OverlapWindow, ResourceShape]] = None
```

### `load_schedule(schedule)`

Maps `partition_key` to the scheduler's current_schedule to get overlap_window and resource_shape.

### `execute(ctx, pre_ctx)`

The core execution method. Replaces the logic from `partition_fuser.py`.

**Execution flow:**

```
1. Communication overlap setup
   - If comm_start == 0: event_record(current_stream)

2. For each comp_op (indexed by fused_idx):
   a. Create OperationContext via ctx.create_op_context(op.operator.op_id)
   b. Read input tensors from ctx.tensor_store using op.input_ports[*].tensor_id
      - Port 0 = main tensor (x)
      - Port 1+ = extra inputs (bias, residual, key, value, rotary, etc.)
   c. Track requires_grad (check x, params, extra_inputs)
   d. If fused_idx == comm_start: launch comm on pre_ctx (OTHER nanobatch)
      - event_wait()
      - Read comm input from pre_ctx.tensor_store
      - comm_op.fuser_forward(...)
   e. Execute: x, extra_outputs = op.operator.fuser_forward(...)
   f. If fused_idx == comm_start - 1: event_record for next comm window
   g. Write output tensors to ctx.tensor_store using op.output_ports[*].tensor_id

3. Handle non-overlapped comm (comm_start == -1 and comm_op != None)
   - event_record → event_wait → fuser_forward → sync

4. Sync comm and write comm outputs to pre_ctx.tensor_store
```

**Key difference from old `partition_fuser.py`:**
- OLD: `isinstance()` checks to determine extra_inputs per op type
- NEW: Port-based routing — read all input_ports, write all output_ports
- OLD: Manual variables (x, bias, residual, key, value)
- NEW: TensorStore with auto-generated tensor_ids

---

## `BackwardPartition`

```python
@dataclass
class BackwardPartition:
    partition_id: int
    partition_key: str
    nano_batch_idx: int
    comp_ops: List[ComputeOp]
    comm_op: Optional[CommunicationOp]
    _schedule_config: Optional[Tuple[OverlapWindow, ResourceShape]] = None
```

### `execute(ctx, pre_ctx) -> Dict[int, List]`

Returns `grad_params: Dict[int, List]` — parameter gradients keyed by op_id.

**Execution flow:**

```
1. Communication overlap setup (same as forward)

2. For each comp_op (indexed by fused_idx):
   a. Retrieve OperationContext via ctx.get_op_context(op.operator.op_id)
      - Contains saved_tensors from forward
   b. If not op_ctx.requires_grad: break
   c. Read grad input tensors from ctx.tensor_store using op.input_ports[*].tensor_id
      - Port 0 = grad of forward output (dx)
      - Port 1+ = grad of extra forward outputs (grad_key, grad_value, etc.)
   d. If fused_idx == comm_start: launch backward comm on pre_ctx
   e. Execute: dx, grad_params, grad_extra = op.operator.fuser_backward(...)
   f. Store grad_params[op.operator.op_id] = grad_params
   g. Free saved_tensors (op_ctx.saved_tensors = None)
   h. Write grad output tensors to ctx.tensor_store using op.output_ports[*].tensor_id

3. Handle non-overlapped backward comm

4. Sync backward comm, write outputs to pre_ctx.tensor_store

5. Return grad_params dict
```

**Key difference from old backward:**
- OLD: `isinstance(op, QKVPostProcessOp)` → `grad_key, grad_value`
- NEW: Read all input_ports, write all output_ports — no isinstance checks
- Backward TensorGraph ports are REVERSED: input_ports = grad of forward output

### Backward channel reversal example (QKVPostProcess):

```
Forward:   input [main]        → output [main, key, value]
Backward:  input [grad_main, grad_key, grad_value]  → output [grad_main]
           (auto-derived by ComputeOpSpec(is_backward=True))
```

---

## Shared Patterns

Both partition classes share:
1. `load_schedule()` — same implementation
2. `get_comm_config()` — same implementation
3. Communication overlap pattern (event_record, event_wait, fuser_forward, sync)
4. ctx/pre_ctx tensor routing pattern

Consider extracting a `_PartitionBase` or common mixin for shared logic, though the plan keeps them separate for clarity.

## Verification Criteria

- ForwardPartition correctly routes tensors through comp_ops via TensorStore
- Communication overlap launches at correct fused_idx
- BackwardPartition retrieves correct OperationContext from forward
- Backward produces grad_params dict with correct op_id keys
- Comm ops read from pre_ctx and write to pre_ctx (OTHER nanobatch)

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 1654-1883 (ForwardPartition) and 1886-2131 (BackwardPartition).
