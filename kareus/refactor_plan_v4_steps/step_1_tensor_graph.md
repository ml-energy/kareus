# Step 1: Core Data Structures — `tensor_graph.py`

## Objective

Create the foundational data structures for the graph-based partition system. This is the **zero-dependency** foundation that all other steps build on.

## File to Create

`kareus/megatron/core/partitions/tensor_graph.py`

Also create `kareus/megatron/core/partitions/__init__.py` (package init).

## Data Structures to Implement

### 1. `CommunicationType` (Enum)

```python
class CommunicationType(Enum):
    ALL_REDUCE = auto()        # 1 input, 1 output (TP)
    ALL_GATHER_KV = auto()     # 2 inputs (k, v), 2 outputs (CP)
    REDUCE_SCATTER_KV = auto() # 2 inputs (grad_k, grad_v), 2 outputs (CP)
```

### 2. `Channel` (Dataclass)

Named connection point on an operator. Maps a semantic name to a positional port index.

```python
@dataclass
class Channel:
    port_idx: int   # positional slot in fuser_forward/fuser_backward
    name: str       # semantic name: "main", "bias", "key", "value", "residual", etc.
```

Key semantics:
- `Channel(0, "main")` — always the primary hidden_states tensor
- Additional channels carry side-channel tensors (key, value, bias, residual)
- Channels **persist** in the TensorGraphBuilder registry until overwritten

### 3. `TensorPort` (Dataclass)

Represents a single input or output port with an auto-generated tensor ID.

```python
@dataclass
class TensorPort:
    port_idx: int
    tensor_id: Optional[str] = None  # e.g., "t_0", "t_1" — assigned by builder
```

### 4. `ComputeOp` (Dataclass)

A computation node in the graph. The operator may be a `PartitionableOperator`
(simple ops) or a `BasicOperation` (from a decomposed `FusedOperation`).

```python
@dataclass
class ComputeOp:
    operator: 'FusibleOperation'  # PartitionableOperator or BasicOperation
    op_id: int = -1  # Unique ID for NanoBatchContext.create/get_op_context
    input_ports: List[TensorPort]
    output_ports: List[TensorPort]

    def get_input_tensor_ids(self) -> List[str]: ...
    def get_output_tensor_ids(self) -> List[str]: ...
```

`op_id` is decoupled from `operator.op_id` because `BasicOperation` (from
TransformerEngine) doesn't have an `op_id` attribute. The `TensorGraphBuilder`
assigns sequential `op_id` values when building `ComputeOp` instances.

### 5. `CommunicationOp` (Dataclass)

A communication node in the graph.

```python
@dataclass
class CommunicationOp:
    comm_type: CommunicationType
    operator: Optional[Any] = None  # Assigned later by transformer_block
    input_ports: List[TensorPort]
    output_ports: List[TensorPort]

    def get_input_tensor_ids(self) -> List[str]: ...
    def get_output_tensor_ids(self) -> List[str]: ...
```

### 6. `PartitionableOperator` (ABC)

Base interface for operators that participate in partitioning.

```python
class PartitionableOperator(ABC):
    def get_forward_ops(self) -> List[Union['ComputeOpSpec', 'CommunicationOpSpec']]: ...

    def get_backward_ops(self) -> List[Union['ComputeOpSpec', 'CommunicationOpSpec']]: ...

    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main")]  # Default: main→main

    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main")]  # Default: main→main
```

Design principles:
- Operators declare FORWARD channels only
- Backward channels are auto-derived by `ComputeOpSpec(is_backward=True)`
- `op_id` lives on `ComputeOp`, assigned by `TensorGraphBuilder` — not on the operator

### 7. `ComputeOpSpec` (Dataclass)

Specification for creating a ComputeOp. Handles automatic backward channel derivation.

When the operator is a `BasicOperation` (from a decomposed `FusedOperation`),
use `input_channels` / `output_channels` to provide channel info explicitly.

```python
@dataclass
class ComputeOpSpec:
    operator: 'FusibleOperation'  # PartitionableOperator or BasicOperation
    is_backward: bool = False
    op_id: Optional[int] = None  # Propagated to ComputeOp.op_id
    input_channels: Optional[List[Channel]] = None   # Override for BasicOperations
    output_channels: Optional[List[Channel]] = None   # Override for BasicOperations

    def get_input_channels(self) -> List[Channel]:
        if self.is_backward:
            fwd_out = self.output_channels or self.operator.get_output_channels()
            return [Channel(i, f"grad_{ch.name}")
                    for i, ch in enumerate(fwd_out)]
        return self.input_channels or self.operator.get_input_channels()

    def get_output_channels(self) -> List[Channel]:
        if self.is_backward:
            fwd_in = self.input_channels or self.operator.get_input_channels()
            return [Channel(i, f"grad_{ch.name}")
                    for i, ch in enumerate(fwd_in)]
        return self.output_channels or self.operator.get_output_channels()
```

FusedOperation decomposition example (Linear = BasicLinear + Bias):
```python
def get_forward_ops(self):
    return [
        ComputeOpSpec(operator=self.basic_ops[0],  # BasicLinear
                      input_channels=[Channel(0, "main")],
                      output_channels=[Channel(0, "main")]),
        ComputeOpSpec(operator=self.basic_ops[1],  # Bias
                      input_channels=[Channel(0, "main")],
                      output_channels=[Channel(0, "main"), Channel(1, "bias")]),
    ]

def get_backward_ops(self):
    # Reversed order
    return [
        ComputeOpSpec(operator=self.basic_ops[1], is_backward=True, ...),  # Bias
        ComputeOpSpec(operator=self.basic_ops[0], is_backward=True, ...),  # BasicLinear
    ]
```

### 8. `CommunicationOpSpec` (Dataclass)

Specification for creating a CommunicationOp. Channels are declared directly (no reversal).

```python
@dataclass
class CommunicationOpSpec:
    comm_type: CommunicationType
    channels: List[Channel]

    def get_input_channels(self) -> List[Channel]:
        return self.channels

    def get_output_channels(self) -> List[Channel]:
        return self.channels  # Comm ops read and write the same channels
```

### 9. `TensorGraphBuilder`

Builds a tensor dependency graph using **named channel routing**.

```python
class TensorGraphBuilder:
    def __init__(self): ...
    def add_initial_channels(self, channels: Dict[str, str]): ...
    def add_op(self, spec: Union[ComputeOpSpec, CommunicationOpSpec]) -> Union[ComputeOp, CommunicationOp]: ...
    def build(self) -> 'TensorGraph': ...
    def get_channel_registry(self) -> Dict[str, str]: ...
```

Key behavior:
- `_channel_registry: Dict[str, str]` maps channel name → tensor_id
- Registry **persists** across operators (NOT cleared after each op)
- `add_op()` wires input ports from registry, creates new tensor IDs for output ports
- Missing channels produce `ext_{name}` tensor IDs (for first-layer None values)
- `_new_tensor_id()` generates `t_0`, `t_1`, `t_2`, ...

### 10. `TensorGraph` (Dataclass)

The built graph.

```python
@dataclass
class TensorGraph:
    ops: List[Union[ComputeOp, CommunicationOp]]
    channel_registry: Dict[str, str]  # Final channel→tensor_id mapping

    def get_compute_ops(self) -> List[ComputeOp]: ...
    def get_comm_ops(self) -> List[CommunicationOp]: ...
    def get_output_channel(self, channel_name: str) -> Optional[str]: ...
```

## Channel Routing Example

For a single layer's attention block:

```
BDA:            reads [main, bias, residual]  → writes [main]
ResidualFork:   reads [main]                  → writes [main, residual]
LN:             reads [main]                  → writes [main]
QKVLinear:      reads [main]                  → writes [main, bias]
QKVPost:        reads [main]                  → writes [main(=query), key, value]
Rotary:         reads [main, key, rotary_pos_emb] → writes [main, key]
CoreAttn:       reads [main, key, value]      → writes [main]
                 ↑ value persists from QKVPost because Rotary doesn't overwrite it
```

## Verification Criteria

- TensorGraphBuilder correctly assigns unique tensor IDs
- Channel persistence works (value from QKVPost reaches CoreAttn through Rotary)
- Missing channels produce `ext_{name}` IDs
- `ComputeOpSpec(is_backward=True)` correctly reverses channel semantics
- `TensorGraph.get_output_channel("main")` returns the last tensor_id

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 717-1056 (Objective 2).
