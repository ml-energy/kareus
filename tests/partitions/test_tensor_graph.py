"""Tests for kareus.megatron.core.partitions.tensor_graph.

Verification criteria from step_1_tensor_graph.md:
  1. TensorGraphBuilder correctly assigns unique tensor IDs
  2. Channel persistence works (value from QKVPost reaches CoreAttn through Rotary)
  3. Missing channels produce ext_{name} IDs
  4. ComputeOpSpec(is_backward=True) correctly reverses channel semantics
  5. TensorGraph.get_output_channel("main") returns the last tensor_id
"""

import pytest

from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    CommunicationOp,
    CommunicationOpSpec,
    CommunicationType,
    ComputeOp,
    ComputeOpSpec,
    PartitionableOperator,
    TensorGraph,
    TensorGraphBuilder,
    TensorPort,
)

# ------------------------------------------------------------------ #
#  Minimal stub operators for testing
# ------------------------------------------------------------------ #


class _StubOp:
    """Minimal stand-in for a BasicOperation (no get_input/output_channels)."""
    pass


class SimpleMainOp(PartitionableOperator):
    """Operator with default main→main channels."""

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class QKVPostOp(PartitionableOperator):
    """Reads main, writes main + key + value."""

    def get_input_channels(self):
        return [Channel(0, "main")]

    def get_output_channels(self):
        return [Channel(0, "main"), Channel(1, "key"), Channel(2, "value")]

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class RotaryOp(PartitionableOperator):
    """Reads main + key + rotary_pos_emb, writes main + key."""

    def get_input_channels(self):
        return [Channel(0, "main"), Channel(1, "key"), Channel(2, "rotary_pos_emb")]

    def get_output_channels(self):
        return [Channel(0, "main"), Channel(1, "key")]

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class CoreAttnOp(PartitionableOperator):
    """Reads main + key + value, writes main."""

    def get_input_channels(self):
        return [Channel(0, "main"), Channel(1, "key"), Channel(2, "value")]

    def get_output_channels(self):
        return [Channel(0, "main")]

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class ResidualForkOp(PartitionableOperator):
    """Reads main, writes main + residual."""

    def get_input_channels(self):
        return [Channel(0, "main")]

    def get_output_channels(self):
        return [Channel(0, "main"), Channel(1, "residual")]

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class BDAOp(PartitionableOperator):
    """Reads main + bias + residual, writes main."""

    def get_input_channels(self):
        return [Channel(0, "main"), Channel(1, "bias"), Channel(2, "residual")]

    def get_output_channels(self):
        return [Channel(0, "main")]

    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]

    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestChannel:
    def test_basic(self):
        ch = Channel(0, "main")
        assert ch.port_idx == 0
        assert ch.name == "main"


class TestTensorPort:
    def test_default_tensor_id_is_none(self):
        port = TensorPort(port_idx=0)
        assert port.tensor_id is None

    def test_explicit_tensor_id(self):
        port = TensorPort(port_idx=1, tensor_id="t_42")
        assert port.tensor_id == "t_42"


class TestComputeOp:
    def test_get_tensor_ids(self):
        op = ComputeOp(
            operator=_StubOp(),
            input_ports=[TensorPort(0, "t_0"), TensorPort(1, "t_1")],
            output_ports=[TensorPort(0, "t_2")],
            op_id=5,
        )
        assert op.get_input_tensor_ids() == ["t_0", "t_1"]
        assert op.get_output_tensor_ids() == ["t_2"]
        assert op.op_id == 5


class TestCommunicationOp:
    def test_get_tensor_ids(self):
        op = CommunicationOp(
            comm_type=CommunicationType.ALL_REDUCE,
            input_ports=[TensorPort(0, "t_0")],
            output_ports=[TensorPort(0, "t_1")],
        )
        assert op.get_input_tensor_ids() == ["t_0"]
        assert op.get_output_tensor_ids() == ["t_1"]

    def test_delegation_to_operator(self):
        """CommunicationOp delegates event_record/event_wait/sync to operator."""

        class MockCommOperator:
            def __init__(self):
                self.calls = []

            def event_record(self, stream):
                self.calls.append(("event_record", stream))

            def event_wait(self):
                self.calls.append(("event_wait",))

            def fuser_forward(self, *args, **kwargs):
                self.calls.append(("fuser_forward",))
                return None, []

            def sync(self, stream):
                self.calls.append(("sync", stream))

        mock = MockCommOperator()
        op = CommunicationOp(
            comm_type=CommunicationType.ALL_REDUCE,
            input_ports=[TensorPort(0, "t_0")],
            output_ports=[TensorPort(0, "t_1")],
            operator=mock,
        )
        op.event_record("stream_0")
        op.event_wait()
        op.fuser_forward([None], None)
        op.sync("stream_0")

        assert len(mock.calls) == 4
        assert mock.calls[0] == ("event_record", "stream_0")
        assert mock.calls[1] == ("event_wait",)
        assert mock.calls[2] == ("fuser_forward",)
        assert mock.calls[3] == ("sync", "stream_0")


class TestComputeOpSpec:
    def test_forward_channels_from_operator(self):
        op = SimpleMainOp()
        spec = ComputeOpSpec(operator=op)
        assert spec.get_input_channels() == [Channel(0, "main")]
        assert spec.get_output_channels() == [Channel(0, "main")]

    def test_forward_channels_from_override(self):
        """BasicOperation-style override with explicit channels."""
        spec = ComputeOpSpec(
            operator=_StubOp(),
            input_channels=[Channel(0, "main")],
            output_channels=[Channel(0, "main"), Channel(1, "bias")],
        )
        assert len(spec.get_input_channels()) == 1
        assert len(spec.get_output_channels()) == 2
        assert spec.get_output_channels()[1].name == "bias"

    def test_backward_reversal(self):
        """is_backward=True reverses channels with grad_ prefix."""
        op = QKVPostOp()
        # Forward: in=[main], out=[main, key, value]
        bwd_spec = ComputeOpSpec(operator=op, is_backward=True)

        # Backward input = grad of forward output
        bwd_in = bwd_spec.get_input_channels()
        assert [ch.name for ch in bwd_in] == ["grad_main", "grad_key", "grad_value"]

        # Backward output = grad of forward input
        bwd_out = bwd_spec.get_output_channels()
        assert [ch.name for ch in bwd_out] == ["grad_main"]

    def test_backward_reversal_with_override(self):
        """Backward with explicit input/output channels (BasicOperation)."""
        spec = ComputeOpSpec(
            operator=_StubOp(),
            is_backward=True,
            input_channels=[Channel(0, "main"), Channel(1, "bias")],
            output_channels=[Channel(0, "main"), Channel(1, "key"), Channel(2, "value")],
        )
        # Backward input = grad of (override) output channels
        bwd_in = spec.get_input_channels()
        assert [ch.name for ch in bwd_in] == ["grad_main", "grad_key", "grad_value"]

        # Backward output = grad of (override) input channels
        bwd_out = spec.get_output_channels()
        assert [ch.name for ch in bwd_out] == ["grad_main", "grad_bias"]


class TestCommunicationOpSpec:
    def test_input_output_same(self):
        spec = CommunicationOpSpec(
            comm_type=CommunicationType.ALL_GATHER_KV,
            channels=[Channel(0, "key"), Channel(1, "value")],
        )
        assert spec.get_input_channels() == spec.get_output_channels()


class TestTensorGraphBuilder:
    def test_unique_tensor_ids(self):
        """Criterion 1: TensorGraphBuilder correctly assigns unique tensor IDs."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        op1 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        op2 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        op3 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))

        # All output tensor IDs should be unique
        all_output_ids = (
            op1.get_output_tensor_ids()
            + op2.get_output_tensor_ids()
            + op3.get_output_tensor_ids()
        )
        assert len(all_output_ids) == len(set(all_output_ids))

        # Sequential naming: t_0, t_1, t_2
        assert all_output_ids == ["t_0", "t_1", "t_2"]

    def test_channel_persistence(self):
        """Criterion 2: value from QKVPost reaches CoreAttn through Rotary.

        QKVPost writes [main, key, value].
        Rotary reads [main, key, rotary_pos_emb] and writes [main, key].
        CoreAttn reads [main, key, value].

        The "value" channel should persist through Rotary (which doesn't
        touch it) so CoreAttn receives the tensor ID from QKVPost.
        """
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        qkv_post = builder.add_op(ComputeOpSpec(operator=QKVPostOp()))
        value_from_qkv = qkv_post.get_output_tensor_ids()[2]  # value port

        rotary = builder.add_op(ComputeOpSpec(operator=RotaryOp()))

        core_attn = builder.add_op(ComputeOpSpec(operator=CoreAttnOp()))
        value_at_core_attn = core_attn.get_input_tensor_ids()[2]  # value port

        # The value tensor_id at CoreAttn must equal the one from QKVPost
        assert value_at_core_attn == value_from_qkv

        # Rotary should NOT have changed the value channel
        # (Rotary only writes main + key, not value)

    def test_missing_channels_produce_ext_ids(self):
        """Criterion 3: Missing channels produce ext_{name} IDs."""
        builder = TensorGraphBuilder()
        # Don't seed any initial channels

        op = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        # "main" was missing → should get "ext_main"
        assert op.get_input_tensor_ids() == ["ext_main"]

    def test_missing_channel_with_multiple_inputs(self):
        """Missing channels each get their own ext_ ID."""
        builder = TensorGraphBuilder()
        # Rotary needs main, key, rotary_pos_emb — none seeded
        op = builder.add_op(ComputeOpSpec(operator=RotaryOp()))
        input_ids = op.get_input_tensor_ids()
        assert input_ids[0] == "ext_main"
        assert input_ids[1] == "ext_key"
        assert input_ids[2] == "ext_rotary_pos_emb"

    def test_get_output_channel(self):
        """Criterion 5: TensorGraph.get_output_channel("main") returns last tensor_id."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))  # writes t_0
        builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))  # writes t_1
        builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))  # writes t_2

        graph = builder.build()
        assert graph.get_output_channel("main") == "t_2"

    def test_sequential_op_ids(self):
        """TensorGraphBuilder assigns sequential op_ids to ComputeOps."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        op1 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        op2 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        op3 = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))

        assert op1.op_id == 0
        assert op2.op_id == 1
        assert op3.op_id == 2

    def test_explicit_op_id(self):
        """ComputeOpSpec with explicit op_id propagates to ComputeOp."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        op = builder.add_op(ComputeOpSpec(operator=SimpleMainOp(), op_id=42))
        assert op.op_id == 42

    def test_communication_op_in_graph(self):
        """CommunicationOpSpec creates a CommunicationOp in the graph."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})

        builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))

        comm_op = builder.add_op(
            CommunicationOpSpec(
                comm_type=CommunicationType.ALL_REDUCE,
                channels=[Channel(0, "main")],
            )
        )
        assert isinstance(comm_op, CommunicationOp)
        assert comm_op.comm_type == CommunicationType.ALL_REDUCE

        # Input reads from registry (t_0 from prev op)
        assert comm_op.get_input_tensor_ids() == ["t_0"]
        # Output gets new tensor_id
        assert comm_op.get_output_tensor_ids() == ["t_1"]

        graph = builder.build()
        assert len(graph.get_compute_ops()) == 1
        assert len(graph.get_comm_ops()) == 1

    def test_full_attention_block_routing(self):
        """End-to-end test matching the channel routing example from the plan.

        BDA:          reads [main, bias, residual] → writes [main]
        ResidualFork: reads [main]                 → writes [main, residual]
        LN:           reads [main]                 → writes [main]
        QKVLinear:    reads [main]                 → writes [main, bias]
        QKVPost:      reads [main]                 → writes [main, key, value]
        Rotary:       reads [main, key, rot_emb]   → writes [main, key]
        CoreAttn:     reads [main, key, value]     → writes [main]
        """
        builder = TensorGraphBuilder()
        builder.add_initial_channels({
            "main": "ext_main",
            "bias": "ext_bias",
            "residual": "ext_residual",
        })

        # BDA
        bda = builder.add_op(ComputeOpSpec(operator=BDAOp()))
        assert bda.get_input_tensor_ids() == ["ext_main", "ext_bias", "ext_residual"]

        # ResidualFork
        res_fork = builder.add_op(ComputeOpSpec(operator=ResidualForkOp()))

        # LN (main→main)
        ln = builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))

        # QKV Linear: reads main, writes main + bias
        # (use explicit channels to simulate a decomposed FusedOperation)
        qkv_linear = builder.add_op(ComputeOpSpec(
            operator=_StubOp(),
            input_channels=[Channel(0, "main")],
            output_channels=[Channel(0, "main"), Channel(1, "bias")],
        ))

        # QKVPost
        qkv_post = builder.add_op(ComputeOpSpec(operator=QKVPostOp()))
        value_tid = qkv_post.get_output_tensor_ids()[2]

        # Rotary — needs rotary_pos_emb which isn't in registry yet
        rotary = builder.add_op(ComputeOpSpec(operator=RotaryOp()))
        assert rotary.get_input_tensor_ids()[2] == "ext_rotary_pos_emb"

        # CoreAttn — should see value from QKVPost (not modified by Rotary)
        core_attn = builder.add_op(ComputeOpSpec(operator=CoreAttnOp()))
        assert core_attn.get_input_tensor_ids()[2] == value_tid

        graph = builder.build()
        assert graph.get_output_channel("main") is not None
        assert len(graph.get_compute_ops()) == 7
        assert len(graph.get_comm_ops()) == 0

    def test_backward_graph(self):
        """Backward graph with is_backward=True reversal."""
        # Build forward graph for a simple chain: SimpleMain → QKVPost
        fwd_builder = TensorGraphBuilder()
        fwd_builder.add_initial_channels({"main": "ext_main"})
        fwd_builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))
        fwd_builder.add_op(ComputeOpSpec(operator=QKVPostOp()))

        # Build backward graph (reversed order)
        bwd_builder = TensorGraphBuilder()
        bwd_builder.add_initial_channels({
            "grad_main": "ext_grad_main",
            "grad_key": "ext_grad_key",
            "grad_value": "ext_grad_value",
        })

        # QKVPost backward: reads grad of [main, key, value], writes grad of [main]
        qkv_bwd = bwd_builder.add_op(
            ComputeOpSpec(operator=QKVPostOp(), is_backward=True)
        )
        assert qkv_bwd.get_input_tensor_ids() == [
            "ext_grad_main", "ext_grad_key", "ext_grad_value",
        ]
        assert len(qkv_bwd.get_output_tensor_ids()) == 1  # grad_main

        # SimpleMain backward: reads grad of [main], writes grad of [main]
        main_bwd = bwd_builder.add_op(
            ComputeOpSpec(operator=SimpleMainOp(), is_backward=True)
        )
        # Should chain from qkv_bwd's output
        assert main_bwd.get_input_tensor_ids() == qkv_bwd.get_output_tensor_ids()

    def test_build_returns_copy(self):
        """build() returns independent lists/dicts."""
        builder = TensorGraphBuilder()
        builder.add_initial_channels({"main": "ext_main"})
        builder.add_op(ComputeOpSpec(operator=SimpleMainOp()))

        graph = builder.build()
        # Mutating the graph shouldn't affect the builder
        graph.ops.clear()
        graph.channel_registry.clear()

        # Builder should still be intact
        reg = builder.get_channel_registry()
        assert "main" in reg


class TestTensorGraph:
    def test_get_output_channel_missing(self):
        graph = TensorGraph()
        assert graph.get_output_channel("nonexistent") is None
