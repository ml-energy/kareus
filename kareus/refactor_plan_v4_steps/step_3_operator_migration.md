# Step 3: Operator Interface Migration

## Objective

Add the `PartitionableOperator` mixin to all existing operator classes. Each operator must declare its channel I/O and return `ComputeOpSpec` / `CommunicationOpSpec` for forward and backward.

## Dependencies

- Step 1: Uses `PartitionableOperator`, `Channel`, `ComputeOpSpec`, `CommunicationOpSpec`, `CommunicationType`

## Files to Modify

All files in `kareus/megatron/core/extensions/ops/`.

## Operator-by-Operator Migration

### 1. `TEColumnParallelLinearOp` (`te_linear.py`)

**Role:** QKV projection (`linear_qkv`), MLP first layer (`linear_fc1`)

**Channels:**
- Input: `[Channel(0, "main")]`
- Output: `[Channel(0, "main")]` or `[Channel(0, "main"), Channel(1, "bias")]` if `return_bias=True`

**Forward ops:** Compute only (no comm — input replicated, output partitioned)
```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]
```

**Backward ops:** Compute → AllReduce (partial sum gradient needs AllReduce)
```python
def get_backward_ops(self):
    return [
        ComputeOpSpec(operator=self, is_backward=True),
        CommunicationOpSpec(
            comm_type=CommunicationType.ALL_REDUCE,
            channels=[Channel(0, "grad_main")],
        ),
    ]
```

**Why backward AllReduce?** Forward `y = x @ W^T` where x is replicated, W is partitioned. Backward `grad_x = grad_y @ W` produces a partial sum → needs AllReduce. This is the partition boundary in backward (mirrors forward's RowParallel AllReduce).

---

### 2. `TERowParallelLinearOp` (`te_linear.py`)

**Role:** Output projection (`linear_proj`), MLP second layer (`linear_fc2`)

**Channels:**
- Input: `[Channel(0, "main")]`
- Output: `[Channel(0, "main")]` or `[Channel(0, "main"), Channel(1, "bias")]` if `return_bias=True`

**Forward ops:** Compute → AllReduce (output is partial sum → needs AllReduce)
```python
def get_forward_ops(self):
    return [
        ComputeOpSpec(operator=self),
        CommunicationOpSpec(
            comm_type=CommunicationType.ALL_REDUCE,
            channels=[Channel(0, "main")],
        ),
    ]
```

**Backward ops:** Compute only (gradient is partitioned, feeds ColumnParallel backward)
```python
def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 3. `ResidualForkOp` (`residual_fork.py`)

**Role:** Fork residual connection before each LayerNorm

**Channels:**
- Input: `[Channel(0, "main")]`
- Output: `[Channel(0, "main"), Channel(1, "residual")]`

**Forward:** `x → (x_main, x_residual)` — identity + copy
**Backward:** `(grad_main, grad_residual) → grad_main + grad_residual` — accumulate at fork

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

Auto-derived backward channels:
- Input: `[Channel(0, "grad_main"), Channel(1, "grad_residual")]`
- Output: `[Channel(0, "grad_main")]`

---

### 4. `TENormOp` / `LayerNormOp` (`te_norm.py`)

**Role:** LayerNorm / RMSNorm

**Channels:** Default `main → main` (no override needed)

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 5. `BiasDropoutAddOp` (`bias_dropout_add.py`)

**Role:** `output = residual + dropout(input + bias)`

**Channels:**
- Input: `[Channel(0, "main"), Channel(1, "bias"), Channel(2, "residual")]`
- Output: `[Channel(0, "main")]`

Bias and residual come from non-adjacent operators via persistent channels.

Auto-derived backward:
- Input: `[Channel(0, "grad_main")]`
- Output: `[Channel(0, "grad_main"), Channel(1, "grad_bias"), Channel(2, "grad_residual")]`

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 6. `QKVPostProcessOp` (`qkv_postprocess.py`)

**Role:** Split mixed_qkv into query, key, value

**Channels:**
- Input: `[Channel(0, "main")]`
- Output: `[Channel(0, "main"), Channel(1, "key"), Channel(2, "value")]`
  - main = query

Auto-derived backward:
- Input: `[Channel(0, "grad_main"), Channel(1, "grad_key"), Channel(2, "grad_value")]`
- Output: `[Channel(0, "grad_main")]`

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 7. `RotaryEmbeddingOp` (`rotary_embedding.py`)

**Role:** Apply RoPE to query and key

**Channels:**
- Input: `[Channel(0, "main"), Channel(1, "key"), Channel(2, "rotary_pos_emb")]`
  - `rotary_pos_emb` is external, seeded via `add_initial_channels()`
- Output: `[Channel(0, "main"), Channel(1, "key")]`
  - Does NOT write `rotary_pos_emb` → channel persists untouched

Auto-derived backward:
- Input: `[Channel(0, "grad_main"), Channel(1, "grad_key")]`
- Output: `[Channel(0, "grad_main"), Channel(1, "grad_key"), Channel(2, "grad_rotary_pos_emb")]`
  - `grad_rotary_pos_emb` produced but unused (no op reads it)

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 8. `TEDotProductAttentionOp` (`te_attention.py`)

**Role:** Core attention computation (query @ key^T → softmax → @ value)

**Channels:**
- Input: `[Channel(0, "main"), Channel(1, "key"), Channel(2, "value")]`
  - key from RotaryEmbed, value from QKVPostProcess (persisted through Rotary!)
- Output: `[Channel(0, "main")]`

**Forward ops (TP only):** Compute only
```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]
```

**Forward ops (CP enabled):** AllGatherKV → Compute
```python
def get_forward_ops(self):
    if self.use_cp:
        return [
            CommunicationOpSpec(
                comm_type=CommunicationType.ALL_GATHER_KV,
                channels=[Channel(0, "key"), Channel(1, "value")],
            ),
            ComputeOpSpec(operator=self),
        ]
    return [ComputeOpSpec(operator=self)]
```

**Backward ops (CP enabled):** AllGatherKV → Compute → ReduceScatterKV
```python
def get_backward_ops(self):
    if self.use_cp:
        return [
            CommunicationOpSpec(
                comm_type=CommunicationType.ALL_GATHER_KV,
                channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
            ),
            ComputeOpSpec(operator=self, is_backward=True),
            CommunicationOpSpec(
                comm_type=CommunicationType.REDUCE_SCATTER_KV,
                channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
            ),
        ]
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

### 9. `BiasSwigluOp` / `BiasGeluOp` / `BiasGegluOp` (`bias_swiglu.py`, `bias_gelu.py`, `bias_geglu.py`)

**Role:** Fused activation with bias (MLP intermediate)

**Channels:**
- Input: `[Channel(0, "main"), Channel(1, "bias")]`
- Output: `[Channel(0, "main")]`

```python
def get_forward_ops(self):
    return [ComputeOpSpec(operator=self)]

def get_backward_ops(self):
    return [ComputeOpSpec(operator=self, is_backward=True)]
```

---

## `__init__.py` Update

Export the new types from the partitions package:
```python
from kareus.megatron.core.partitions.tensor_graph import (
    PartitionableOperator, Channel, ComputeOpSpec, CommunicationOpSpec, CommunicationType,
)
```

## Implementation Notes

- Each operator class gains the `PartitionableOperator` mixin via multiple inheritance
- Existing `FusibleOperation` inheritance stays — both interfaces coexist
- `op_id` is NOT set in `__init__`; it's assigned later by `_build_partitions`
- The `use_cp` flag on `TEDotProductAttentionOp` determines whether CP communication ops are included

## Verification Criteria

- Each operator's `get_forward_ops()` and `get_backward_ops()` return valid specs
- Building a TensorGraph from a full layer's operators produces correct channel routing
- Channel persistence verified: value from QKVPost reaches CoreAttn through Rotary
- Backward channel auto-derivation produces correct grad_ prefixed channels

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 1059-1377 (Objective 2, operator examples).
