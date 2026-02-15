# Step 5: Partition Builder — `partition_builder.py`

## Objective

Create `PartitionBuilder` — the component that automatically forms interleaved nanobatch partitions from already-built TensorGraphs.

## Dependencies

- Step 1: Uses `TensorGraph`, `ComputeOp`, `CommunicationOp`
- Step 4: Uses `ForwardPartition`, `BackwardPartition`

## File to Create

`kareus/megatron/core/partitions/partition_builder.py`

## Design

### `PartitionBuilder`

```python
class PartitionBuilder:
    def __init__(
        self,
        forward_tensor_graph: TensorGraph,
        backward_tensor_graph: TensorGraph,
        config: TransformerConfig,
    ): ...

    def build_forward_partitions(self) -> List[ForwardPartition]: ...
    def build_backward_partitions(self) -> List[BackwardPartition]: ...
```

Takes SEPARATE forward and backward TensorGraphs (already built with correct port connections) and forms partition instances.

---

### Core Algorithm: `_form_partitions()`

**Input:** A flat list of ops `[A, AR, B, AR, C]` where A/B/C are ComputeOps and AR is CommunicationOp.

**Step 1: Split by communication boundaries** (`_split_by_communications`)

```
Input:  [A, AR, B, C, AR, D]
Output: [([A], AR), ([B, C], AR), ([D], None)]
```

Each segment = (list of compute ops, trailing comm op or None).

**Step 2: Interleave nanobatches**

For 2 nanobatches, the interleaved partition sequence is:

```
Given segments: [seg0=(A, AR), seg1=(B, AR)]

Partitions formed:
  P0: NB1, comp=[A], comm=None       ← first partition has no comm to wait for
  P1: NB2, comp=[A], comm=AR_seg0    ← NB2 waits for NB1's AR from seg0
  P2: NB1, comp=[B], comm=AR_seg0    ← NB1 waits for NB2's AR from seg0
  P3: NB2, comp=[B], comm=AR_seg1    ← NB2 waits for NB1's AR from seg1
```

Key insight:
- NB1 partitions wait for NB2's comm from the **previous** segment
- NB2 partitions wait for NB1's comm from the **current** segment (just started)

**Pseudocode:**

```python
def _form_partitions(self, ops, partition_class):
    segments = self._split_by_communications(ops)
    partitions = []
    prev_comm = None

    for seg_idx, (comp_ops, comm_after) in enumerate(segments):
        # NB1 partition: compute + wait for NB2's previous comm
        partitions.append(partition_class(
            partition_id=len(partitions),
            nano_batch_idx=0,
            comp_ops=comp_ops,
            comm_op=prev_comm,
        ))

        # NB2 partition: compute + wait for NB1's current comm
        partitions.append(partition_class(
            partition_id=len(partitions),
            nano_batch_idx=1,
            comp_ops=comp_ops,
            comm_op=comm_after,
        ))

        prev_comm = comm_after  # NB2's comm, for NB1 in next iteration

    return partitions
```

---

### `_split_by_communications()`

```python
def _split_by_communications(self, ops):
    """
    Split ops into segments by communication boundaries.

    Input: [A, AR, B, C, AR, D]
    Output: [([A], AR), ([B, C], AR), ([D], None)]
    """
    segments = []
    current_comp_ops = []

    for op in ops:
        if isinstance(op, ComputeOp):
            current_comp_ops.append(op)
        elif isinstance(op, CommunicationOp):
            if current_comp_ops:
                segments.append((current_comp_ops, op))
                current_comp_ops = []
            else:
                # Comm at start: attach to previous segment or handle edge case
                if segments:
                    segments[-1] = (segments[-1][0], op)

    # Trailing compute ops (no comm after)
    if current_comp_ops:
        segments.append((current_comp_ops, None))

    return segments
```

---

### `partition_key` Assignment

Each partition needs a `partition_key` string for scheduler lookup. This maps to attributes on the schedule object (e.g., `schedule.fwd_attn_nb0`, `schedule.bwd_mlp_nb1`).

The naming scheme should encode:
- Direction: `fwd` / `bwd`
- Segment index or semantic name
- Nanobatch index: `nb0` / `nb1`

Exact naming depends on how the scheduler exposes its schedule items.

---

## Concrete Example: Single Layer Forward

A single LLaMA layer produces this forward op sequence:

```
BDA → ResidualFork → LN → QKVLinear → QKVPost → Rotary → CoreAttn → OProj(+AR) →
BDA → ResidualFork → LN → FC1 → BiasSwiglu → FC2(+AR)
```

Split by AR communication:
```
Segment 0: [BDA, ResidualFork, LN, QKVLinear, QKVPost, Rotary, CoreAttn, OProj], comm=AR
Segment 1: [BDA, ResidualFork, LN, FC1, BiasSwiglu, FC2], comm=AR
```

Partitions formed:
```
P0: NB1, comp=[BDA..OProj], comm=None         (first, no comm to wait)
P1: NB2, comp=[BDA..OProj], comm=AR_seg0      (wait NB1's AR)
P2: NB1, comp=[BDA..FC2],   comm=AR_seg0      (wait NB2's AR from seg0)
P3: NB2, comp=[BDA..FC2],   comm=AR_seg1      (wait NB1's AR from seg1)
```

For N layers, the same pattern repeats N times, giving 4N partitions.

---

## Backward Partition Formation

The backward TensorGraph has ops in **reverse order** with **different communication** (ColumnParallel backward has AllReduce, RowParallel backward does not).

Backward op sequence (single layer, reversed):
```
FC2_bwd → FC1_bwd(+AR) → BiasSwiglu_bwd → LN_bwd → ResidualFork_bwd → BDA_bwd →
OProj_bwd → CoreAttn_bwd → Rotary_bwd → QKVPost_bwd → QKVLinear_bwd(+AR) → LN_bwd → ResidualFork_bwd → BDA_bwd
```

Split by AR gives backward segments/partitions following the same algorithm.

## Verification Criteria

- `_split_by_communications` correctly splits at CommunicationOp boundaries
- Forward partitions for a single layer produce 4 partitions (2 segments x 2 NB)
- Multi-layer produces 4N partitions
- `comm_op` assignment is correct: NB1 gets prev_comm, NB2 gets current comm
- First partition always has `comm_op=None`
- Backward partitions form correctly with reversed ops

## Code Reference (from plan)

See `kareus/refactor_plan_v4.md` lines 1382-1529 (Objective 3).
