# Step 2: Context Management — `context_manager.py`

## Objective

Create `TensorStore` and `NanoBatchContext` — the runtime containers that hold tensors and operation contexts during forward/backward execution.

## Dependencies

- Step 1: Uses `TensorPort` from `tensor_graph.py`

## File to Create

`kareus/megatron/core/partitions/context_manager.py`

## Data Structures to Implement

### 1. `TensorStore`

Stores tensors by their auto-generated tensor IDs (from TensorGraphBuilder).

```python
@dataclass
class TensorStore:
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def set(self, tensor_id: str, tensor: torch.Tensor): ...
    def get(self, tensor_id: str) -> Optional[torch.Tensor]: ...
    def get_by_ports(self, ports: List[TensorPort]) -> List[torch.Tensor]: ...
    def set_from_ports(self, ports: List[TensorPort], tensors: List[torch.Tensor]): ...
```

Key behavior:
- Simple dict-based storage with `tensor_id → Tensor` mapping
- `get_by_ports()` reads multiple tensors at once for a list of TensorPorts
- `set_from_ports()` writes multiple tensors at once, skipping None values

### 2. `NanoBatchContext`

Context for a single nano-batch across all partitions. Holds both the tensor store and per-operator saved state for backward.

```python
@dataclass
class NanoBatchContext:
    batch_idx: int  # 0 or 1

    # Tensor storage — tensors stored by auto-generated IDs (t_0, t_1, ...)
    tensor_store: TensorStore = field(default_factory=TensorStore)

    # Operation contexts for backward, keyed by op_id (ComputeOp.op_id)
    op_contexts: Dict[int, OperationContext] = field(default_factory=dict)

    # Bookkeeping for save_for_backward flattening
    _saved_tensors: List[torch.Tensor] = field(default_factory=list)
    _saved_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
```

Methods:

#### `create_op_context(op_id: int) -> OperationContext`

Called during **forward**. Creates a new `OperationContext` from TransformerEngine for saving activations needed by backward.

- `op_id` comes from `ComputeOp.op_id`, assigned by `TensorGraphBuilder`
- Stored in `self.op_contexts[op_id]`

#### `get_op_context(op_id: int) -> OperationContext`

Called during **backward**. Retrieves the context saved during forward.

- Same `op_id` works because forward and backward ComputeOps share the same operator instance

#### `flatten_saved_tensors() -> List[torch.Tensor]`

Called after all forward partitions execute. Flattens all `OperationContext.to_save` into a single list for `save_for_backward()`.

- Iterates `op_contexts` in op_id order
- For each context: records `(range_start, range_end)` in `_saved_ranges`
- Clears `ctx.to_save = None` after extraction

#### `restore_saved_tensors(saved_tensors: Tuple[torch.Tensor, ...])`

Called at the start of backward. Restores saved tensors back into each OperationContext.

- Uses `_saved_ranges` to slice the flat tensor tuple
- Sets `ctx.saved_tensors = saved_tensors[start:end]` for each op_id

## Cross-NanoBatch Pattern (ctx vs pre_ctx)

```
┌──────────────────────────────────────────────────────────────────────┐
│  ctx      = NanoBatchContext for THIS nanobatch                     │
│  pre_ctx  = NanoBatchContext for the OTHER nanobatch                │
│                                                                      │
│  COMPUTE OPS: read/write tensors via ctx.tensor_store                │
│  COMM OP: read/write tensors via pre_ctx.tensor_store                │
│                                                                      │
│  This means the comm op processes the OTHER nanobatch's data         │
│  while compute ops process THIS nanobatch's data concurrently.       │
└──────────────────────────────────────────────────────────────────────┘
```

## External Dependency

Uses `OperationContext` from TransformerEngine:
```python
from transformer_engine.pytorch.ops.op import OperationContext
```

This class provides:
- `to_save: Optional[List[torch.Tensor]]` — tensors to save during forward
- `saved_tensors: Optional[Tuple[torch.Tensor, ...]]` — restored for backward
- `requires_grad: bool` — whether gradients are needed

## Verification Criteria

- TensorStore correctly stores and retrieves by tensor_id
- `get_by_ports()` returns tensors in port order
- `flatten_saved_tensors()` → `restore_saved_tensors()` round-trips correctly
- op_id keying works: forward create → backward get returns same context

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 1548-1652 (Objective 4, context_manager section).
