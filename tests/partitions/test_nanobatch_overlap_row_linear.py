#!/usr/bin/env python
"""Validate nanobatch overlap path with two row-linear layers.

This script compares:
1) Nanobatch partition execution using:
   - graph build logic equivalent to TransformerBlock._build_partitions
   - TransformerBlockAutogradFunction
2) Direct sequential execution on a full batch:
   - layer2(all_reduce(layer1(x))) called directly

Decoupled from Megatron: uses only torch.distributed + KareusAllReduce.

Expected usage:
    torchrun --nproc_per_node=2 tests/partitions/test_nanobatch_overlap_row_linear.py
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from kareus.transformer_engine.pytorch.ops import Linear as KareusLinear  # noqa: E402
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce as KareusAllReduce  # noqa: E402
from kareus.megatron.core.partitions import (  # noqa: E402
    CommunicationType,
    ComputeOp,
    PartitionBuilder,
    SeedConfig,
    TensorGraphBuilder,
    TransformerBlockAutogradFunction,
)
from kareus.megatron.core.partitions.tensor_graph import (  # noqa: E402
    Channel,
    CommunicationOp,
    CommunicationOpSpec,
    ComputeOpSpec,
    PartitionableOperator,
)


# ---------------------------------------------------------------------------
#  Standalone row-linear op (no Megatron dependencies)
# ---------------------------------------------------------------------------


class RowLinearOp(KareusLinear, PartitionableOperator):
    """Row-parallel-style linear layer decoupled from Megatron.

    Wraps KareusLinear and implements PartitionableOperator so it can
    participate in partition graph builds.  When ``trailing_allreduce``
    is True the forward graph declares a trailing ALL_REDUCE communication
    op, mirroring TERowParallelLinearOp.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device=None,
        dtype=None,
        num_batches: int = 2,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        trailing_allreduce: bool = True,
    ):
        KareusLinear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            return_bias=False,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=1,
            sequence_parallel=False,
            rng_state_tracker_function=None,
            accumulate_into_main_grad=False,
            use_persistent_output=False,
            num_batches=num_batches,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        self._trailing_allreduce = trailing_allreduce

    # -- PartitionableOperator interface --

    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main")]

    def get_forward_ops(self):
        ops = [ComputeOpSpec(operator=self)]
        if self._trailing_allreduce:
            ops.append(CommunicationOpSpec(
                comm_type=CommunicationType.ALL_REDUCE,
                channels=[Channel(0, "main")],
            ))
        return ops

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]

    def forward(self, x, batch_idx=0):
        output = KareusLinear.forward(self, x, batch_idx=batch_idx)
        if isinstance(output, tuple):
            return output
        return output, None


# ---------------------------------------------------------------------------
#  Autograd-compatible all-reduce for the baseline path
# ---------------------------------------------------------------------------


class _AllReduceFunc(torch.autograd.Function):
    """Differentiable all-reduce: forward sums across ranks, backward is
    identity (gradients are already identical across ranks)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup):
        ctx.group = group
        out = input_.clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def allreduce_autograd(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return _AllReduceFunc.apply(x, group)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _verify_partition_formation(nano_block: "TwoLayerRowLinearNanoBatchBlock", rank: int) -> None:
    """Verify the partition formation matches expected structure.

    Expected forward graph: ComputeOp(L1) -> CommunicationOp(AR) -> ComputeOp(L2)
    Expected segments: [([L1], AR), ([L2], None)]
    Expected partitions:
      P0 (fwd_seg0_nb0): nb=0, comp=[L1], comm=None
      P1 (fwd_seg0_nb1): nb=1, comp=[L1], comm=AR  (AllReduces NB0's output)
      P2 (fwd_seg1_nb0): nb=0, comp=[L2], comm=AR  (AllReduces NB1's output)
      P3 (fwd_seg1_nb1): nb=1, comp=[L2], comm=None
    """
    fwd_graph = nano_block.forward_tensor_graph
    fwd_parts = nano_block.forward_partitions

    # --- 1. Check forward graph ops ---
    ops = fwd_graph.ops
    assert len(ops) == 3, f"Expected 3 forward graph ops, got {len(ops)}"
    assert isinstance(ops[0], ComputeOp), f"ops[0] should be ComputeOp, got {type(ops[0]).__name__}"
    assert isinstance(ops[1], CommunicationOp), f"ops[1] should be CommunicationOp, got {type(ops[1]).__name__}"
    assert isinstance(ops[2], ComputeOp), f"ops[2] should be ComputeOp, got {type(ops[2]).__name__}"

    # Check operator identity
    assert ops[0].operator is nano_block.layers[0], "ops[0].operator should be Layer1"
    assert ops[2].operator is nano_block.layers[1], "ops[2].operator should be Layer2"
    assert ops[1].comm_type == CommunicationType.ALL_REDUCE, "ops[1] should be ALL_REDUCE"

    # Check op_ids
    assert ops[0].op_id == 0, f"Layer1 op_id should be 0, got {ops[0].op_id}"
    assert ops[2].op_id == 1, f"Layer2 op_id should be 1, got {ops[2].op_id}"

    # --- 2. Check tensor wiring ---
    # L1: ext_main -> t_0
    assert ops[0].input_ports[0].tensor_id == "ext_main", \
        f"L1 input should be ext_main, got {ops[0].input_ports[0].tensor_id}"
    assert ops[0].output_ports[0].tensor_id == "t_0", \
        f"L1 output should be t_0, got {ops[0].output_ports[0].tensor_id}"

    # AR: t_0 -> t_1
    assert ops[1].input_ports[0].tensor_id == "t_0", \
        f"AR input should be t_0, got {ops[1].input_ports[0].tensor_id}"
    assert ops[1].output_ports[0].tensor_id == "t_1", \
        f"AR output should be t_1, got {ops[1].output_ports[0].tensor_id}"

    # L2: t_1 -> t_2
    assert ops[2].input_ports[0].tensor_id == "t_1", \
        f"L2 input should be t_1, got {ops[2].input_ports[0].tensor_id}"
    assert ops[2].output_ports[0].tensor_id == "t_2", \
        f"L2 output should be t_2, got {ops[2].output_ports[0].tensor_id}"

    # Final output channel
    assert fwd_graph.get_output_channel("main") == "t_2", \
        f"Final 'main' channel should be t_2, got {fwd_graph.get_output_channel('main')}"

    # --- 3. Check segment splitting ---
    segments = PartitionBuilder._split_by_communications(ops)
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"

    seg0_comp, seg0_comm = segments[0]
    seg1_comp, seg1_comm = segments[1]

    assert len(seg0_comp) == 1 and seg0_comp[0].operator is nano_block.layers[0], \
        "Segment 0 compute should be [Layer1]"
    assert seg0_comm is not None and seg0_comm.comm_type == CommunicationType.ALL_REDUCE, \
        "Segment 0 comm should be ALL_REDUCE"
    assert len(seg1_comp) == 1 and seg1_comp[0].operator is nano_block.layers[1], \
        "Segment 1 compute should be [Layer2]"
    assert seg1_comm is None, \
        f"Segment 1 comm should be None, got {seg1_comm}"

    # --- 4. Check partition formation ---
    assert len(fwd_parts) == 4, f"Expected 4 forward partitions, got {len(fwd_parts)}"

    p0, p1, p2, p3 = fwd_parts

    # P0: fwd_seg0_nb0, nb=0, comp=[L1], comm=None
    assert p0.partition_key == "fwd_seg0_nb0", f"P0 key: {p0.partition_key}"
    assert p0.nano_batch_idx == 0, f"P0 nb: {p0.nano_batch_idx}"
    assert len(p0.comp_ops) == 1, f"P0 should have 1 comp_op, got {len(p0.comp_ops)}"
    assert p0.comp_ops[0].operator is nano_block.layers[0], "P0 comp should be Layer1"
    assert p0.comm_op is None, f"P0 comm should be None, got {p0.comm_op}"

    # P1: fwd_seg0_nb1, nb=1, comp=[L1], comm=AR (AllReduces NB0's L1 output)
    assert p1.partition_key == "fwd_seg0_nb1", f"P1 key: {p1.partition_key}"
    assert p1.nano_batch_idx == 1, f"P1 nb: {p1.nano_batch_idx}"
    assert len(p1.comp_ops) == 1, f"P1 should have 1 comp_op, got {len(p1.comp_ops)}"
    assert p1.comp_ops[0].operator is nano_block.layers[0], "P1 comp should be Layer1"
    assert p1.comm_op is not None, "P1 should have a comm_op"
    assert p1.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P1 comm should be ALL_REDUCE"
    # P1's comm reads t_0 (NB0's L1 output) and writes t_1
    assert p1.comm_op.input_ports[0].tensor_id == "t_0", \
        f"P1 comm input should be t_0, got {p1.comm_op.input_ports[0].tensor_id}"
    assert p1.comm_op.output_ports[0].tensor_id == "t_1", \
        f"P1 comm output should be t_1, got {p1.comm_op.output_ports[0].tensor_id}"

    # P2: fwd_seg1_nb0, nb=0, comp=[L2], comm=AR (AllReduces NB1's L1 output)
    assert p2.partition_key == "fwd_seg1_nb0", f"P2 key: {p2.partition_key}"
    assert p2.nano_batch_idx == 0, f"P2 nb: {p2.nano_batch_idx}"
    assert len(p2.comp_ops) == 1, f"P2 should have 1 comp_op, got {len(p2.comp_ops)}"
    assert p2.comp_ops[0].operator is nano_block.layers[1], "P2 comp should be Layer2"
    assert p2.comm_op is not None, "P2 should have a comm_op"
    assert p2.comm_op.comm_type == CommunicationType.ALL_REDUCE, "P2 comm should be ALL_REDUCE"
    # P2's comm reads t_0 (NB1's L1 output) and writes t_1
    assert p2.comm_op.input_ports[0].tensor_id == "t_0", \
        f"P2 comm input should be t_0, got {p2.comm_op.input_ports[0].tensor_id}"
    assert p2.comm_op.output_ports[0].tensor_id == "t_1", \
        f"P2 comm output should be t_1, got {p2.comm_op.output_ports[0].tensor_id}"

    # P3: fwd_seg1_nb1, nb=1, comp=[L2], comm=None
    assert p3.partition_key == "fwd_seg1_nb1", f"P3 key: {p3.partition_key}"
    assert p3.nano_batch_idx == 1, f"P3 nb: {p3.nano_batch_idx}"
    assert len(p3.comp_ops) == 1, f"P3 should have 1 comp_op, got {len(p3.comp_ops)}"
    assert p3.comp_ops[0].operator is nano_block.layers[1], "P3 comp should be Layer2"
    assert p3.comm_op is None, f"P3 comm should be None, got {p3.comm_op}"

    # --- 5. Check AllReduce operator assignment ---
    # P1 and P2 should have physical AllReduce operators assigned (not None)
    assert p1.comm_op.operator is not None, "P1 comm_op.operator should be assigned"
    assert p2.comm_op.operator is not None, "P2 comm_op.operator should be assigned"
    # They should be different operator instances (one per nanobatch)
    assert p1.comm_op.operator is not p2.comm_op.operator, \
        "P1 and P2 should have different AllReduce operator instances"
    # P1 (nb=1) gets comm_ops[1], P2 (nb=0) gets comm_ops[0]
    assert p1.nano_batch_idx == 1, "P1 is nb1"
    assert p2.nano_batch_idx == 0, "P2 is nb0"

    if rank == 0:
        print("[PARTITION CHECK] Forward graph: OK (3 ops: ComputeOp → CommunicationOp → ComputeOp)")
        print("[PARTITION CHECK] Tensor wiring: OK (ext_main → t_0 → t_1 → t_2)")
        print("[PARTITION CHECK] Segments: OK (2 segments: [L1,AR], [L2,None])")
        print("[PARTITION CHECK] Partitions: OK (4 partitions with correct nb/comp/comm)")
        print(f"  P0: {p0.partition_key} nb={p0.nano_batch_idx} comp=[L1] comm=None")
        print(f"  P1: {p1.partition_key} nb={p1.nano_batch_idx} comp=[L1] comm=AR(t_0→t_1)")
        print(f"  P2: {p2.partition_key} nb={p2.nano_batch_idx} comp=[L2] comm=AR(t_0→t_1)")
        print(f"  P3: {p3.partition_key} nb={p3.nano_batch_idx} comp=[L2] comm=None")
        print("[PARTITION CHECK] AllReduce assignment: OK (distinct operators per nanobatch)")


def _verify_execution_behavior(
    nano_block: "TwoLayerRowLinearNanoBatchBlock",
    ref_layers: List[RowLinearOp],
    x: torch.Tensor,
    pg: dist.ProcessGroup,
    rank: int,
) -> None:
    """Verify step-by-step partition execution matches expected behavior.

    Expected execution for 2 nanobatches (NB0, NB1):
      Step 1 - P0: L1(NB0)                           [no comm]
      Step 2 - P1: L1(NB1)  overlapped with AR(NB0)  [comm on NB0]
      Step 3 - P2: L2(NB0)  overlapped with AR(NB1)  [comm on NB1]
      Step 4 - P3: L2(NB1)                           [no comm]

    Computes ground-truth intermediates per nanobatch using ref_layers,
    then manually steps through partitions and compares tensor store values.
    """
    from kareus.megatron.core.partitions.context_manager import NanoBatchContext

    mid = x.size(1) // 2
    x_nb0 = x[:, :mid, :].detach().clone()
    x_nb1 = x[:, mid:, :].detach().clone()

    # --- Compute reference intermediates per nanobatch ---
    with torch.no_grad():
        ref_l1_nb0, _ = ref_layers[0](x_nb0, batch_idx=0)
        ref_l1_nb1, _ = ref_layers[0](x_nb1, batch_idx=1)

        ref_ar_nb0 = ref_l1_nb0.clone()
        dist.all_reduce(ref_ar_nb0, group=pg)
        ref_ar_nb1 = ref_l1_nb1.clone()
        dist.all_reduce(ref_ar_nb1, group=pg)

        ref_l2_nb0, _ = ref_layers[1](ref_ar_nb0, batch_idx=0)
        ref_l2_nb1, _ = ref_layers[1](ref_ar_nb1, batch_idx=1)

    # --- Manually step through partitions (mirrors autograd_function.forward) ---
    ctx_nb0 = NanoBatchContext(batch_idx=0)
    ctx_nb1 = NanoBatchContext(batch_idx=1)
    ctx_nb0.tensor_store.set("ext_main", x_nb0)
    ctx_nb1.tensor_store.set("ext_main", x_nb1)

    partitions = nano_block.forward_partitions
    for p in partitions:
        p.is_grad_enabled = False

    checks_passed = True

    def _check(label, actual, expected):
        nonlocal checks_passed
        diff = (actual - expected).abs().max().item()
        ok = diff < 1e-3
        if not ok:
            checks_passed = False
        if rank == 0:
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {label}: max_diff={diff:.6e}")

    # Step 1: P0 (fwd_seg0_nb0) — L1(NB0), no comm
    partitions[0].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
    if rank == 0:
        print("Step 1 - P0: L1(NB0), no comm")
    _check("NB0 t_0 vs ref_L1(NB0)", ctx_nb0.tensor_store.get("t_0"), ref_l1_nb0)

    # Step 2: P1 (fwd_seg0_nb1) — L1(NB1) overlapped with AR(NB0)
    partitions[1].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
    if rank == 0:
        print("Step 2 - P1: L1(NB1) + AR(NB0)")
    _check("NB1 t_0 vs ref_L1(NB1)", ctx_nb1.tensor_store.get("t_0"), ref_l1_nb1)
    _check("NB0 t_1 vs ref_AR(NB0)", ctx_nb0.tensor_store.get("t_1"), ref_ar_nb0)

    # Step 3: P2 (fwd_seg1_nb0) — L2(NB0) overlapped with AR(NB1)
    partitions[2].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
    if rank == 0:
        print("Step 3 - P2: L2(NB0) + AR(NB1)")
    _check("NB0 t_2 vs ref_L2(NB0)", ctx_nb0.tensor_store.get("t_2"), ref_l2_nb0)
    _check("NB1 t_1 vs ref_AR(NB1)", ctx_nb1.tensor_store.get("t_1"), ref_ar_nb1)

    # Step 4: P3 (fwd_seg1_nb1) — L2(NB1), no comm
    partitions[3].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
    if rank == 0:
        print("Step 4 - P3: L2(NB1), no comm")
    _check("NB1 t_2 vs ref_L2(NB1)", ctx_nb1.tensor_store.get("t_2"), ref_l2_nb1)

    if rank == 0:
        if checks_passed:
            print("[EXECUTION CHECK] All intermediate values match reference!")
        else:
            print("[EXECUTION CHECK] FAILED — some intermediate values don't match!")


def _verify_autograd_function(
    nano_block: "TwoLayerRowLinearNanoBatchBlock",
    ref_layers: List[RowLinearOp],
    x: torch.Tensor,
    pg: dist.ProcessGroup,
    rank: int,
) -> None:
    """Verify TransformerBlockAutogradFunction behavior step-by-step.

    Tests 3 hypotheses to narrow down the forward output mismatch:

    H1: Does is_grad_enabled=True (vs False) change partition execution results?
        Runs partitions manually with is_grad_enabled=True + detached inputs,
        wrapped in torch.no_grad to mimic autograd.Function context.
        Compares against half-batch reference.

    H2: Does full-batch reference match concatenated split-batch references?
        Compares ref_layers[0](x_full)[:,:mid,:] vs ref_layers[0](x_nb0).
        If these differ, the main test's comparison of full-batch ref vs
        split-batch test output is invalid.

    H3: Exact autograd function replication (requires_grad + no_grad context)
        Uses requires_grad=True view inputs (like TransformerBlockAutogradFunction
        receives) and wraps in torch.no_grad to mimic autograd.Function context.
        Tests whether requires_grad on inputs causes the divergence.
    """
    from kareus.megatron.core.partitions.context_manager import NanoBatchContext

    mid = x.size(1) // 2

    # --- Compute per-nanobatch reference (half-batch, via ref_layers) ---
    x_nb0 = x[:, :mid, :].detach().clone()
    x_nb1 = x[:, mid:, :].detach().clone()

    with torch.no_grad():
        ref_l1_nb0, _ = ref_layers[0](x_nb0, batch_idx=0)
        ref_l1_nb1, _ = ref_layers[0](x_nb1, batch_idx=1)

        ref_ar_nb0 = ref_l1_nb0.clone()
        dist.all_reduce(ref_ar_nb0, group=pg)
        ref_ar_nb1 = ref_l1_nb1.clone()
        dist.all_reduce(ref_ar_nb1, group=pg)

        ref_l2_nb0, _ = ref_layers[1](ref_ar_nb0, batch_idx=0)
        ref_l2_nb1, _ = ref_layers[1](ref_ar_nb1, batch_idx=1)

    partitions = nano_block.forward_partitions

    checks = {"H1": True, "H2": True, "H3": True}

    def _check(hyp, label, actual, expected, atol=1e-3):
        diff = (actual - expected).abs().max().item()
        ok = diff < atol
        if not ok:
            checks[hyp] = False
        if rank == 0:
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {label}: max_diff={diff:.6e}")

    # ================================================================
    # H2: Full-batch vs split-batch consistency
    # ================================================================
    if rank == 0:
        print(f"\n=== H2: Full-batch vs split-batch consistency ===")

    with torch.no_grad():
        full_out_l1, _ = ref_layers[0](x)
        _check("H2", "L1: full[:,:mid,:] vs split_nb0",
               full_out_l1[:, :mid, :], ref_l1_nb0)
        _check("H2", "L1: full[:,mid:,:] vs split_nb1",
               full_out_l1[:, mid:, :], ref_l1_nb1)

        # Also check after all-reduce and L2
        full_ar = full_out_l1.clone()
        dist.all_reduce(full_ar, group=pg)
        _check("H2", "AR: full[:,:mid,:] vs split_nb0",
               full_ar[:, :mid, :], ref_ar_nb0)
        _check("H2", "AR: full[:,mid:,:] vs split_nb1",
               full_ar[:, mid:, :], ref_ar_nb1)

        full_out_l2, _ = ref_layers[1](full_ar)
        _check("H2", "L2: full[:,:mid,:] vs split_nb0",
               full_out_l2[:, :mid, :], ref_l2_nb0)
        _check("H2", "L2: full[:,mid:,:] vs split_nb1",
               full_out_l2[:, mid:, :], ref_l2_nb1)

    if rank == 0:
        status = "PASS" if checks["H2"] else "FAIL"
        print(f"  H2 Result: {status}")

    # ================================================================
    # H1: Partitions with is_grad_enabled=True, detached inputs
    #     (inside no_grad to mimic autograd.Function context)
    # ================================================================
    if rank == 0:
        print(f"\n=== H1: Partitions with is_grad_enabled=True (detached inputs) ===")

    ctx_nb0 = NanoBatchContext(batch_idx=0)
    ctx_nb1 = NanoBatchContext(batch_idx=1)
    ctx_nb0.tensor_store.set("ext_main", x_nb0.detach().clone())
    ctx_nb1.tensor_store.set("ext_main", x_nb1.detach().clone())

    for p in partitions:
        p.is_grad_enabled = True

    with torch.no_grad():
        partitions[0].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
        _check("H1", "P0: NB0 t_0", ctx_nb0.tensor_store.get("t_0"), ref_l1_nb0)

        partitions[1].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
        _check("H1", "P1: NB1 t_0", ctx_nb1.tensor_store.get("t_0"), ref_l1_nb1)
        _check("H1", "P1: NB0 t_1 (AR)",
               ctx_nb0.tensor_store.get("t_1"), ref_ar_nb0)

        partitions[2].execute(ctx=ctx_nb0, pre_ctx=ctx_nb1)
        _check("H1", "P2: NB0 t_2", ctx_nb0.tensor_store.get("t_2"), ref_l2_nb0)
        _check("H1", "P2: NB1 t_1 (AR)",
               ctx_nb1.tensor_store.get("t_1"), ref_ar_nb1)

        partitions[3].execute(ctx=ctx_nb1, pre_ctx=ctx_nb0)
        _check("H1", "P3: NB1 t_2", ctx_nb1.tensor_store.get("t_2"), ref_l2_nb1)

    if rank == 0:
        status = "PASS" if checks["H1"] else "FAIL"
        print(f"  H1 Result: {status}")

    # ================================================================
    # H3: Autograd function replication (requires_grad + no_grad context)
    #     Uses view inputs with requires_grad=True, exactly like
    #     TransformerBlockAutogradFunction.forward receives.
    # ================================================================
    if rank == 0:
        print(f"\n=== H3: Autograd function replication (requires_grad=True, no_grad) ===")

    x_clone = x.detach().clone().requires_grad_(True)
    h1 = x_clone[:, :mid, :]
    h2 = x_clone[:, mid:, :]

    ctx_nb0_h3 = NanoBatchContext(batch_idx=0)
    ctx_nb1_h3 = NanoBatchContext(batch_idx=1)
    ctx_nb0_h3.tensor_store.set("ext_main", h1)
    ctx_nb1_h3.tensor_store.set("ext_main", h2)

    for p in partitions:
        p.is_grad_enabled = True

    with torch.no_grad():
        partitions[0].execute(ctx=ctx_nb0_h3, pre_ctx=ctx_nb1_h3)
        _check("H3", "P0: NB0 t_0",
               ctx_nb0_h3.tensor_store.get("t_0"), ref_l1_nb0)

        partitions[1].execute(ctx=ctx_nb1_h3, pre_ctx=ctx_nb0_h3)
        _check("H3", "P1: NB1 t_0",
               ctx_nb1_h3.tensor_store.get("t_0"), ref_l1_nb1)
        _check("H3", "P1: NB0 t_1 (AR)",
               ctx_nb0_h3.tensor_store.get("t_1"), ref_ar_nb0)

        partitions[2].execute(ctx=ctx_nb0_h3, pre_ctx=ctx_nb1_h3)
        _check("H3", "P2: NB0 t_2",
               ctx_nb0_h3.tensor_store.get("t_2"), ref_l2_nb0)
        _check("H3", "P2: NB1 t_1 (AR)",
               ctx_nb1_h3.tensor_store.get("t_1"), ref_ar_nb1)

        partitions[3].execute(ctx=ctx_nb1_h3, pre_ctx=ctx_nb0_h3)
        _check("H3", "P3: NB1 t_2",
               ctx_nb1_h3.tensor_store.get("t_2"), ref_l2_nb1)

    if rank == 0:
        status = "PASS" if checks["H3"] else "FAIL"
        print(f"  H3 Result: {status}")

    # ================================================================
    # H3b: Same as H3 but with .contiguous() on the view inputs
    #       Tests whether non-contiguous views are the root cause.
    # ================================================================
    checks["H3b"] = True

    if rank == 0:
        print(f"\n=== H3b: Autograd replication (requires_grad=True, CONTIGUOUS) ===")

    x_clone2 = x.detach().clone().requires_grad_(True)
    h1c = x_clone2[:, :mid, :].contiguous()
    h2c = x_clone2[:, mid:, :].contiguous()

    ctx_nb0_h3b = NanoBatchContext(batch_idx=0)
    ctx_nb1_h3b = NanoBatchContext(batch_idx=1)
    ctx_nb0_h3b.tensor_store.set("ext_main", h1c)
    ctx_nb1_h3b.tensor_store.set("ext_main", h2c)

    for p in partitions:
        p.is_grad_enabled = True

    with torch.no_grad():
        partitions[0].execute(ctx=ctx_nb0_h3b, pre_ctx=ctx_nb1_h3b)
        _check("H3b", "P0: NB0 t_0",
               ctx_nb0_h3b.tensor_store.get("t_0"), ref_l1_nb0)

        partitions[1].execute(ctx=ctx_nb1_h3b, pre_ctx=ctx_nb0_h3b)
        _check("H3b", "P1: NB1 t_0",
               ctx_nb1_h3b.tensor_store.get("t_0"), ref_l1_nb1)
        _check("H3b", "P1: NB0 t_1 (AR)",
               ctx_nb0_h3b.tensor_store.get("t_1"), ref_ar_nb0)

        partitions[2].execute(ctx=ctx_nb0_h3b, pre_ctx=ctx_nb1_h3b)
        _check("H3b", "P2: NB0 t_2",
               ctx_nb0_h3b.tensor_store.get("t_2"), ref_l2_nb0)
        _check("H3b", "P2: NB1 t_1 (AR)",
               ctx_nb1_h3b.tensor_store.get("t_1"), ref_ar_nb1)

        partitions[3].execute(ctx=ctx_nb1_h3b, pre_ctx=ctx_nb0_h3b)
        _check("H3b", "P3: NB1 t_2",
               ctx_nb1_h3b.tensor_store.get("t_2"), ref_l2_nb1)

    if rank == 0:
        status = "PASS" if checks["H3b"] else "FAIL"
        print(f"  H3b Result: {status}")

    # ================================================================
    # Summary
    # ================================================================
    if rank == 0:
        print(f"\n=== Autograd Verification Summary ===")
        for hyp, ok in checks.items():
            print(f"  {hyp}: {'PASS' if ok else 'FAIL'}")
        if not checks["H3"] and checks["H3b"]:
            print("  ROOT CAUSE: Non-contiguous view inputs cause incorrect"
                  " GEMM results. Fix: add .contiguous() when splitting"
                  " nanobatches.")
        elif all(checks.values()):
            print("  All hypotheses pass → bug is in torch.autograd.Function"
                  " context or state corruption from prior verification runs")
        elif not checks["H2"]:
            print("  H2 FAIL → full-batch != split-batch reference."
                  " Main test comparison is invalid!")
        elif not checks["H1"]:
            print("  H1 FAIL → is_grad_enabled=True changes execution results")
        elif not checks["H3"]:
            print("  H3 FAIL → requires_grad inputs cause the divergence")


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max_diff={_max_diff(a, b):.6e}, "
            f"a_shape={tuple(a.shape)}, b_shape={tuple(b.shape)}"
        )


def _init_distributed() -> Tuple[int, int, int]:
    """Lightweight distributed init using only torch.distributed."""
    if not dist.is_initialized():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
#  Nanobatch block (same graph/autograd path as TransformerBlock)
# ---------------------------------------------------------------------------


class TwoLayerRowLinearNanoBatchBlock(torch.nn.Module):
    """Two-layer block using the same graph/autograd nanobatch path as TransformerBlock."""

    def __init__(self, layers: List[RowLinearOp], process_group: dist.ProcessGroup):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self._process_group = process_group
        self._build_partitions()
        self._assign_allreduce_comm_ops()

    def _build_partitions(self) -> None:
        all_ops = list(self.layers)

        fwd_builder = TensorGraphBuilder()
        fwd_builder.add_initial_channels({"main": "ext_main"})

        fwd_op_id_map = {}
        for op in all_ops:
            for spec in op.get_forward_ops():
                concrete_op = fwd_builder.add_op(spec)
                if isinstance(concrete_op, ComputeOp):
                    fwd_op_id_map[id(spec.operator)] = concrete_op.op_id
        self.forward_tensor_graph = fwd_builder.build()

        bwd_builder = TensorGraphBuilder()
        bwd_builder.add_initial_channels({"grad_main": "ext_grad_main"})
        for op in reversed(all_ops):
            for spec in op.get_backward_ops():
                if isinstance(spec, ComputeOpSpec) and spec.op_id is None:
                    fwd_id = fwd_op_id_map.get(id(spec.operator))
                    if fwd_id is not None:
                        spec.op_id = fwd_id
                bwd_builder.add_op(spec)
        self.backward_tensor_graph = bwd_builder.build()

        builder = PartitionBuilder(
            forward_graph=self.forward_tensor_graph,
            backward_graph=self.backward_tensor_graph,
        )
        self.forward_partitions = builder.build_forward_partitions()
        self.backward_partitions = builder.build_backward_partitions()
        self.seed_config = SeedConfig(rotary_tid=None, mask_tid=None)

    def _assign_allreduce_comm_ops(self) -> None:
        pg = self._process_group
        comm_ops = [
            KareusAllReduce(process_group=pg, async_op=True, backend="nccl")
            for _ in range(2)
        ]
        for partition in self.forward_partitions + self.backward_partitions:
            if (
                partition.comm_op is not None
                and partition.comm_op.comm_type == CommunicationType.ALL_REDUCE
            ):
                partition.comm_op.operator = comm_ops[partition.nano_batch_idx]

    def _get_all_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for op in self.layers:
            params.extend(list(op.parameters()))
        return params

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(1)
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even for 2-way nanobatch split, got {batch_size}")
        mid_point = batch_size // 2
        h1 = hidden_states[:, :mid_point, ...]
        h2 = hidden_states[:, mid_point:, ...]

        all_params = self._get_all_params()
        is_grad_enabled = torch.is_grad_enabled()
        h1_out, h2_out = TransformerBlockAutogradFunction.apply(
            h1,
            h2,
            None,  # rotary_pos_emb
            None,  # attention_mask
            self.forward_partitions,
            self.backward_partitions,
            self.forward_tensor_graph,
            self.backward_tensor_graph,
            None,  # scheduler
            None,  # config
            self.seed_config,
            is_grad_enabled,
            *all_params,
        )
        return torch.cat([h1_out, h2_out], dim=1)


# ---------------------------------------------------------------------------
#  Layer construction
# ---------------------------------------------------------------------------


def _build_row_linear_pair(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    device,
    dtype,
) -> List[RowLinearOp]:
    nano_batch_size = batch_size // 2

    layer1 = RowLinearOp(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=True,
        device=device,
        dtype=dtype,
        num_batches=2,
        batch_size=nano_batch_size,
        seq_length=seq_len,
        trailing_allreduce=True,
    )

    layer2 = RowLinearOp(
        in_features=hidden_size,
        out_features=hidden_size,
        bias=True,
        device=device,
        dtype=dtype,
        num_batches=2,
        batch_size=nano_batch_size,
        seq_length=seq_len,
        trailing_allreduce=False,
    )

    with torch.no_grad():
        torch.nn.init.normal_(layer1.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(layer2.weight, mean=0.0, std=0.02)
        if layer1.bias is not None:
            layer1.bias.zero_()
        if layer2.bias is not None:
            layer2.bias.zero_()

    return [layer1, layer2]


# ---------------------------------------------------------------------------
#  Test runner
# ---------------------------------------------------------------------------


def run_case(args: argparse.Namespace) -> None:
    rank, world_size, _ = _init_distributed()
    if args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be even for two nanobatches")

    pg = dist.distributed_c10d._get_default_group()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    # Seed all ranks identically so weights & input are the same everywhere.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ref_layers = _build_row_linear_pair(
        args.hidden_size, args.batch_size, args.seq_len, device, dtype,
    )
    test_layers = _build_row_linear_pair(
        args.hidden_size, args.batch_size, args.seq_len, device, dtype,
    )

    with torch.no_grad():
        for p_ref, p_test in zip(
            list(ref_layers[0].parameters()) + list(ref_layers[1].parameters()),
            list(test_layers[0].parameters()) + list(test_layers[1].parameters()),
        ):
            p_test.copy_(p_ref)

    torch.manual_seed(args.seed + 100)
    x = torch.randn(
        args.seq_len,
        args.batch_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    x_ref = x.detach().clone().requires_grad_(True)
    x_test = x.detach().clone().requires_grad_(True)

    # Baseline: direct sequential execution with explicit all-reduce
    # between layers (matching the partition graph which has AR after layer1).
    y_ref, _ = ref_layers[0](x_ref)
    y_ref = allreduce_autograd(y_ref, pg)
    y_ref, _ = ref_layers[1](y_ref)
    loss_ref = y_ref.float().pow(2).mean()
    loss_ref.backward()

    # Nanobatch partition path (layer1 → AR → layer2, via partitions).
    nano_block = TwoLayerRowLinearNanoBatchBlock(test_layers, pg).to(device)

    # Verify partition formation before running
    _verify_partition_formation(nano_block, rank)

    # Verify autograd function behavior (Tests H1, H2, H3)
    _verify_autograd_function(nano_block, ref_layers, x, pg, rank)

    # Skip old execution verification to test state corruption (H4)
    # Uncomment to re-enable:
    # _verify_execution_behavior(nano_block, ref_layers, x, pg, rank)

    y_test = nano_block(x_test)
    loss_test = y_test.float().pow(2).mean()
    loss_test.backward()

    atol = args.atol
    rtol = args.rtol

    # Per-nanobatch diagnostics
    mid = args.batch_size // 2
    if rank == 0:
        print(f"\n--- Forward output diagnostics ---")
        print(f"  Full output diff: {_max_diff(y_test, y_ref):.6e}")
        print(f"  NB0 diff (y_test[:,:mid] vs y_ref[:,:mid]): {_max_diff(y_test[:, :mid, :], y_ref[:, :mid, :]):.6e}")
        print(f"  NB1 diff (y_test[:,mid:] vs y_ref[:,mid:]): {_max_diff(y_test[:, mid:, :], y_ref[:, mid:, :]):.6e}")
        # Check if test output looks like a scaling of ref (e.g. 2x from double AR)
        ratio_nb0 = (y_test[:, :mid, :].float() / y_ref[:, :mid, :].float().clamp(min=1e-6)).mean().item()
        ratio_nb1 = (y_test[:, mid:, :].float() / y_ref[:, mid:, :].float().clamp(min=1e-6)).mean().item()
        print(f"  NB0 mean ratio (test/ref): {ratio_nb0:.6f}")
        print(f"  NB1 mean ratio (test/ref): {ratio_nb1:.6f}")
        # Check norms
        print(f"  y_ref  norm: {y_ref.float().norm():.4f}, NB0: {y_ref[:,:mid,:].float().norm():.4f}, NB1: {y_ref[:,mid:,:].float().norm():.4f}")
        print(f"  y_test norm: {y_test.float().norm():.4f}, NB0: {y_test[:,:mid,:].float().norm():.4f}, NB1: {y_test[:,mid:,:].float().norm():.4f}")

    _assert_close("forward_output", y_test, y_ref, atol=atol, rtol=rtol)
    _assert_close("input_grad", x_test.grad, x_ref.grad, atol=atol, rtol=rtol)

    for idx, (ref_op, test_op) in enumerate(zip(ref_layers, test_layers), start=1):
        _assert_close(
            f"layer{idx}.weight_grad",
            test_op.weight.grad,
            ref_op.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        if ref_op.bias is not None:
            _assert_close(
                f"layer{idx}.bias_grad",
                test_op.bias.grad,
                ref_op.bias.grad,
                atol=atol,
                rtol=rtol,
            )

    dist.barrier()
    if rank == 0:
        print(
            "[PASS] Nanobatch overlap path matches direct sequential RowLinearOp "
            f"(seq={args.seq_len}, batch={args.batch_size}, hidden={args.hidden_size}, "
            f"world_size={world_size})"
        )


def _cleanup() -> None:
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_case(args)
    finally:
        _cleanup()
