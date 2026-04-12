"""Tests for the partition system with a single Linear operator.

Verifies that the graph-based partition system produces correct results
by comparing with standard PyTorch Linear execution:

1. Graph construction — tensor IDs and channel routing
2. Partition formation — partition count and structure
3. Forward correctness — output matches F.linear
4. Backward correctness — gradients match F.linear
"""

import pytest
import torch
import torch.nn.functional as F
from typing import List, Tuple

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext

from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    ComputeOp,
    ComputeOpSpec,
    PartitionableOperator,
    TensorGraph,
    TensorGraphBuilder,
)
from kareus.megatron.core.partitions.partition_builder import PartitionBuilder
from kareus.megatron.core.partitions.autograd_function import (
    SeedConfig,
    TransformerBlockAutogradFunction,
)


# ------------------------------------------------------------------ #
#  LinearOp — minimal PartitionableOperator wrapping nn.Linear
# ------------------------------------------------------------------ #


class LinearOp(BasicOperation, PartitionableOperator):
    """Minimal Linear operator for testing the partition system.

    Implements the fuser_forward/fuser_backward interface required by
    ForwardPartition and BackwardPartition.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self._has_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_params()

    def _reset_params(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self._has_bias:
            torch.nn.init.zeros_(self.bias)

    # -- BasicOperation abstract methods (not used directly, routed via fuser_*) --

    def op_forward(self, ctx, input_, **kwargs):
        raise NotImplementedError("Use fuser_forward")

    def op_backward(self, ctx, grad_output):
        raise NotImplementedError("Use fuser_backward")

    # -- FusibleOperation interface --

    def fuser_forward(
        self,
        basic_op_ctxs,
        input_,
        *,
        basic_op_extra_inputs,
        basic_op_prev_ops,
        basic_op_next_ops,
        basic_op_kwargs,
    ):
        ctx = basic_op_ctxs[0]
        ctx.save_for_backward(input_)
        output = F.linear(input_, self.weight, self.bias)
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs,
        grad_output,
        *,
        basic_op_grad_extra_outputs,
    ):
        ctx = basic_op_ctxs[0]
        (input_,) = ctx.saved_tensors

        # grad w.r.t. input
        grad_input = grad_output @ self.weight

        # grad w.r.t. parameters
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input_.reshape(-1, input_.shape[-1])
        grad_weight = grad_output_2d.t() @ input_2d

        if self._has_bias:
            grad_bias = grad_output_2d.sum(dim=0)
            return grad_input, [(grad_weight, grad_bias)], [()]
        else:
            return grad_input, [(grad_weight,)], [()]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _build_graphs(op: LinearOp) -> Tuple[TensorGraph, TensorGraph]:
    """Build forward and backward TensorGraphs for a single operator.

    Propagates forward op_ids to backward specs so that backward
    partitions can retrieve the correct OperationContext.
    """
    # Forward graph
    fwd_builder = TensorGraphBuilder()
    fwd_builder.add_initial_channels({"main": "ext_main"})
    fwd_op_id_map = {}
    for spec in op.get_forward_ops():
        concrete_op = fwd_builder.add_op(spec)
        if isinstance(concrete_op, ComputeOp):
            fwd_op_id_map[id(spec.operator)] = concrete_op.op_id
    fwd_graph = fwd_builder.build()

    # Backward graph — assign matching op_ids
    bwd_builder = TensorGraphBuilder()
    bwd_builder.add_initial_channels({"grad_main": "ext_grad_main"})
    for spec in op.get_backward_ops():
        if isinstance(spec, ComputeOpSpec):
            spec.op_id = fwd_op_id_map.get(id(spec.operator))
        bwd_builder.add_op(spec)
    bwd_graph = bwd_builder.build()

    return fwd_graph, bwd_graph


def _build_partitions(op: LinearOp):
    """Build graphs, partitions, and seed config for a single operator."""
    fwd_graph, bwd_graph = _build_graphs(op)
    pb = PartitionBuilder(fwd_graph, bwd_graph)
    fwd_parts = pb.build_forward_partitions()
    bwd_parts = pb.build_backward_partitions()
    seed_config = SeedConfig(
        h_tid="ext_main",
        rotary_tid=None,
        mask_tid=None,
        fwd_output_channel="main",
        bwd_grad_tid="ext_grad_main",
        bwd_output_channel="grad_main",
    )
    return fwd_graph, bwd_graph, fwd_parts, bwd_parts, seed_config


# ------------------------------------------------------------------ #
#  Tests — Graph Construction (CPU)
# ------------------------------------------------------------------ #


class TestGraphConstruction:
    """Verify TensorGraph structure for a single Linear operator."""

    def test_forward_graph_structure(self):
        """Forward graph: 1 ComputeOp, ext_main → t_0."""
        op = LinearOp(16, 32)
        fwd_graph, _ = _build_graphs(op)

        compute_ops = fwd_graph.get_compute_ops()
        assert len(compute_ops) == 1
        assert len(fwd_graph.get_comm_ops()) == 0

        cop = compute_ops[0]
        assert cop.op_id == 0
        assert cop.get_input_tensor_ids() == ["ext_main"]
        assert cop.get_output_tensor_ids() == ["t_0"]

    def test_forward_output_channel(self):
        """Final 'main' channel points to t_0."""
        op = LinearOp(16, 32)
        fwd_graph, _ = _build_graphs(op)
        assert fwd_graph.get_output_channel("main") == "t_0"

    def test_backward_graph_structure(self):
        """Backward graph: grad_main → grad_main, matching forward op_id."""
        op = LinearOp(16, 32)
        _, bwd_graph = _build_graphs(op)

        compute_ops = bwd_graph.get_compute_ops()
        assert len(compute_ops) == 1

        cop = compute_ops[0]
        assert cop.op_id == 0  # matches forward op_id
        assert cop.get_input_tensor_ids() == ["ext_grad_main"]
        assert cop.get_output_tensor_ids() == ["t_0"]

    def test_backward_output_channel(self):
        """Backward 'grad_main' channel points to t_0."""
        op = LinearOp(16, 32)
        _, bwd_graph = _build_graphs(op)
        assert bwd_graph.get_output_channel("grad_main") == "t_0"


# ------------------------------------------------------------------ #
#  Tests — Partition Formation (CPU)
# ------------------------------------------------------------------ #


class TestPartitionFormation:
    """Verify PartitionBuilder output for a single Linear operator."""

    def test_partition_count(self):
        """No comm ops → 1 segment → 2 partitions per direction."""
        op = LinearOp(16, 32)
        _, _, fwd_parts, bwd_parts, _ = _build_partitions(op)
        assert len(fwd_parts) == 2
        assert len(bwd_parts) == 2

    def test_nanobatch_assignment(self):
        """Partitions alternate NB0 and NB1."""
        op = LinearOp(16, 32)
        _, _, fwd_parts, bwd_parts, _ = _build_partitions(op)

        assert fwd_parts[0].nano_batch_idx == 0
        assert fwd_parts[1].nano_batch_idx == 1
        assert bwd_parts[0].nano_batch_idx == 0
        assert bwd_parts[1].nano_batch_idx == 1

    def test_no_comm_ops(self):
        """All partitions have comm_op=None (no communication)."""
        op = LinearOp(16, 32)
        _, _, fwd_parts, bwd_parts, _ = _build_partitions(op)

        for p in fwd_parts + bwd_parts:
            assert p.comm_op is None

    def test_partitions_share_compute_ops(self):
        """Both NB0 and NB1 partitions reference the same compute ops."""
        op = LinearOp(16, 32)
        _, _, fwd_parts, _, _ = _build_partitions(op)

        assert fwd_parts[0].comp_ops is fwd_parts[1].comp_ops
        assert len(fwd_parts[0].comp_ops) == 1

    def test_partition_keys(self):
        """Partition keys follow fwd_seg{i}_nb{j} naming."""
        op = LinearOp(16, 32)
        _, _, fwd_parts, bwd_parts, _ = _build_partitions(op)

        assert fwd_parts[0].partition_key == "fwd_seg0_nb0"
        assert fwd_parts[1].partition_key == "fwd_seg0_nb1"
        assert bwd_parts[0].partition_key == "bwd_seg0_nb0"
        assert bwd_parts[1].partition_key == "bwd_seg0_nb1"


# ------------------------------------------------------------------ #
#  Tests — Forward/Backward Correctness (CUDA required)
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestForwardBackwardCorrectness:
    """Numerical correctness: partition execution vs standard PyTorch."""

    @staticmethod
    def _run_partition(op, h1, h2, fwd_graph, bwd_graph, fwd_parts, bwd_parts, seed_config):
        """Run TransformerBlockAutogradFunction and return (h1_out, h2_out)."""
        # Capture grad mode before entering the autograd Function,
        # mirroring the pattern in transformer_block.py.
        is_grad_enabled = torch.is_grad_enabled()
        params = list(op.parameters())
        return TransformerBlockAutogradFunction.apply(
            h1, h2, None, None,
            fwd_parts, bwd_parts,
            fwd_graph, bwd_graph,
            None, None, seed_config,
            is_grad_enabled,
            *params,
        )

    def test_forward_output(self):
        """Partition forward output matches F.linear for each nanobatch."""
        torch.manual_seed(42)
        op = LinearOp(16, 32).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
        h2 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)

        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)

        # Reference
        out1_ref = F.linear(h1.detach(), op.weight.detach(), op.bias.detach())
        out2_ref = F.linear(h2.detach(), op.weight.detach(), op.bias.detach())

        torch.testing.assert_close(h1_out, out1_ref)
        torch.testing.assert_close(h2_out, out2_ref)

    def test_backward_input_grads(self):
        """Input gradients from backward match reference."""
        torch.manual_seed(42)
        op = LinearOp(16, 32).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
        h2 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)

        # Reference (separate computation graph)
        h1_ref = h1.detach().clone().requires_grad_(True)
        h2_ref = h2.detach().clone().requires_grad_(True)
        W = op.weight.detach().clone()
        b = op.bias.detach().clone()

        out1_ref = F.linear(h1_ref, W, b)
        out2_ref = F.linear(h2_ref, W, b)
        (out1_ref.sum() + out2_ref.sum()).backward()

        # Partition
        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)
        (h1_out.sum() + h2_out.sum()).backward()

        torch.testing.assert_close(h1.grad, h1_ref.grad)
        torch.testing.assert_close(h2.grad, h2_ref.grad)

    def test_backward_param_grads(self):
        """Parameter gradients (weight, bias) match reference.

        The partition system splits the batch into two nanobatches and
        _combine_param_grads sums their gradients.  This must equal the
        gradient from processing both nanobatches with the same params.
        """
        torch.manual_seed(42)
        op = LinearOp(16, 32).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
        h2 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)

        # Reference with separate param copies
        W_ref = op.weight.detach().clone().requires_grad_(True)
        b_ref = op.bias.detach().clone().requires_grad_(True)
        h1_d = h1.detach().clone()
        h2_d = h2.detach().clone()

        out1_ref = F.linear(h1_d, W_ref, b_ref)
        out2_ref = F.linear(h2_d, W_ref, b_ref)
        (out1_ref.sum() + out2_ref.sum()).backward()

        # Partition
        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)
        (h1_out.sum() + h2_out.sum()).backward()

        torch.testing.assert_close(op.weight.grad, W_ref.grad)
        torch.testing.assert_close(op.bias.grad, b_ref.grad)

    def test_no_bias(self):
        """Correctness with bias=False."""
        torch.manual_seed(42)
        op = LinearOp(16, 32, bias=False).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
        h2 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)

        # Reference
        W_ref = op.weight.detach().clone().requires_grad_(True)
        h1_ref = h1.detach().clone().requires_grad_(True)
        h2_ref = h2.detach().clone().requires_grad_(True)

        out1_ref = F.linear(h1_ref, W_ref)
        out2_ref = F.linear(h2_ref, W_ref)
        (out1_ref.sum() + out2_ref.sum()).backward()

        # Partition
        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)
        (h1_out.sum() + h2_out.sum()).backward()

        torch.testing.assert_close(h1_out.detach(), out1_ref.detach())
        torch.testing.assert_close(h2_out.detach(), out2_ref.detach())
        torch.testing.assert_close(h1.grad, h1_ref.grad)
        torch.testing.assert_close(h2.grad, h2_ref.grad)
        torch.testing.assert_close(op.weight.grad, W_ref.grad)

    def test_larger_dimensions(self):
        """Correctness with larger batch/seq/hidden dimensions."""
        torch.manual_seed(42)
        D_in, D_out = 128, 256
        op = LinearOp(D_in, D_out).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(4, 64, D_in, device="cuda", requires_grad=True)
        h2 = torch.randn(4, 64, D_in, device="cuda", requires_grad=True)

        # Reference
        W_ref = op.weight.detach().clone().requires_grad_(True)
        b_ref = op.bias.detach().clone().requires_grad_(True)
        h1_ref = h1.detach().clone().requires_grad_(True)
        h2_ref = h2.detach().clone().requires_grad_(True)

        out1_ref = F.linear(h1_ref, W_ref, b_ref)
        out2_ref = F.linear(h2_ref, W_ref, b_ref)
        (out1_ref.sum() + out2_ref.sum()).backward()

        # Partition
        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)
        (h1_out.sum() + h2_out.sum()).backward()

        torch.testing.assert_close(h1_out.detach(), out1_ref.detach())
        torch.testing.assert_close(h2_out.detach(), out2_ref.detach())
        torch.testing.assert_close(h1.grad, h1_ref.grad)
        torch.testing.assert_close(h2.grad, h2_ref.grad)
        torch.testing.assert_close(op.weight.grad, W_ref.grad)
        torch.testing.assert_close(op.bias.grad, b_ref.grad)

    def test_nonuniform_grad(self):
        """Correctness with non-uniform (non-ones) upstream gradient.

        Uses a random external gradient instead of .sum() to exercise
        the full gradient path.
        """
        torch.manual_seed(42)
        op = LinearOp(16, 32).cuda()
        fwd_g, bwd_g, fwd_p, bwd_p, sc = _build_partitions(op)

        h1 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)
        h2 = torch.randn(2, 8, 16, device="cuda", requires_grad=True)

        # Reference
        W_ref = op.weight.detach().clone().requires_grad_(True)
        b_ref = op.bias.detach().clone().requires_grad_(True)
        h1_ref = h1.detach().clone().requires_grad_(True)
        h2_ref = h2.detach().clone().requires_grad_(True)

        out1_ref = F.linear(h1_ref, W_ref, b_ref)
        out2_ref = F.linear(h2_ref, W_ref, b_ref)

        # Partition
        h1_out, h2_out = self._run_partition(op, h1, h2, fwd_g, bwd_g, fwd_p, bwd_p, sc)

        # Random upstream gradients (same for both)
        torch.manual_seed(999)
        g1 = torch.randn_like(h1_out)
        g2 = torch.randn_like(h2_out)

        torch.autograd.backward([out1_ref, out2_ref], [g1, g2])
        torch.autograd.backward([h1_out, h2_out], [g1, g2])

        torch.testing.assert_close(h1.grad, h1_ref.grad)
        torch.testing.assert_close(h2.grad, h2_ref.grad)
        torch.testing.assert_close(op.weight.grad, W_ref.grad)
        torch.testing.assert_close(op.bias.grad, b_ref.grad)
