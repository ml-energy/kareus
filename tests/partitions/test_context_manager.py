"""Tests for kareus.megatron.core.partitions.context_manager.

Verification criteria from step_2_context_manager.md:
  1. TensorStore correctly stores and retrieves by tensor_id
  2. get_by_ports() returns tensors in port order
  3. flatten_saved_tensors() → restore_saved_tensors() round-trips correctly
  4. op_id keying works: forward create → backward get returns same context
"""

import torch
import pytest

from kareus.megatron.core.partitions.context_manager import (
    NanoBatchContext,
    TensorStore,
)
from kareus.megatron.core.partitions.tensor_graph import TensorPort


# ------------------------------------------------------------------ #
#  TensorStore tests
# ------------------------------------------------------------------ #


class TestTensorStore:
    def test_set_and_get(self):
        """Criterion 1: correctly stores and retrieves by tensor_id."""
        store = TensorStore()
        t = torch.tensor([1.0, 2.0, 3.0])
        store.set("t_0", t)
        assert torch.equal(store.get("t_0"), t)

    def test_get_missing_returns_none(self):
        store = TensorStore()
        assert store.get("nonexistent") is None

    def test_overwrite(self):
        store = TensorStore()
        store.set("t_0", torch.tensor([1.0]))
        store.set("t_0", torch.tensor([2.0]))
        assert torch.equal(store.get("t_0"), torch.tensor([2.0]))

    def test_get_by_ports(self):
        """Criterion 2: get_by_ports returns tensors in port order."""
        store = TensorStore()
        t0 = torch.tensor([10.0])
        t1 = torch.tensor([20.0])
        t2 = torch.tensor([30.0])
        store.set("t_0", t0)
        store.set("t_1", t1)
        store.set("t_2", t2)

        ports = [
            TensorPort(port_idx=0, tensor_id="t_2"),
            TensorPort(port_idx=1, tensor_id="t_0"),
            TensorPort(port_idx=2, tensor_id="t_1"),
        ]
        result = store.get_by_ports(ports)
        assert torch.equal(result[0], t2)
        assert torch.equal(result[1], t0)
        assert torch.equal(result[2], t1)

    def test_get_by_ports_missing(self):
        """get_by_ports returns None for missing tensor_ids."""
        store = TensorStore()
        store.set("t_0", torch.tensor([1.0]))
        ports = [
            TensorPort(port_idx=0, tensor_id="t_0"),
            TensorPort(port_idx=1, tensor_id="t_999"),
        ]
        result = store.get_by_ports(ports)
        assert torch.equal(result[0], torch.tensor([1.0]))
        assert result[1] is None

    def test_set_from_ports(self):
        store = TensorStore()
        ports = [
            TensorPort(port_idx=0, tensor_id="t_0"),
            TensorPort(port_idx=1, tensor_id="t_1"),
            TensorPort(port_idx=2, tensor_id="t_2"),
        ]
        tensors = [
            torch.tensor([10.0]),
            torch.tensor([20.0]),
            torch.tensor([30.0]),
        ]
        store.set_from_ports(ports, tensors)
        assert torch.equal(store.get("t_0"), tensors[0])
        assert torch.equal(store.get("t_1"), tensors[1])
        assert torch.equal(store.get("t_2"), tensors[2])

    def test_set_from_ports_skips_none(self):
        store = TensorStore()
        ports = [
            TensorPort(port_idx=0, tensor_id="t_0"),
            TensorPort(port_idx=1, tensor_id="t_1"),
        ]
        tensors = [torch.tensor([10.0]), None]
        store.set_from_ports(ports, tensors)
        assert torch.equal(store.get("t_0"), torch.tensor([10.0]))
        assert store.get("t_1") is None


# ------------------------------------------------------------------ #
#  NanoBatchContext tests
# ------------------------------------------------------------------ #


class TestNanoBatchContext:
    def test_create_and_get_op_context(self):
        """Criterion 4: op_id keying — forward create → backward get."""
        ctx = NanoBatchContext(batch_idx=0)

        op_ctx_0 = ctx.create_op_context(0)
        op_ctx_1 = ctx.create_op_context(1)

        # get returns the same object
        assert ctx.get_op_context(0) is op_ctx_0
        assert ctx.get_op_context(1) is op_ctx_1
        assert ctx.get_op_context(0) is not ctx.get_op_context(1)

    def test_op_context_has_correct_defaults(self):
        ctx = NanoBatchContext(batch_idx=0)
        op_ctx = ctx.create_op_context(42)
        assert op_ctx.requires_grad is True
        assert op_ctx.saved_tensors is None
        assert op_ctx.to_save is None

    def test_get_missing_op_context_raises(self):
        ctx = NanoBatchContext(batch_idx=0)
        with pytest.raises(KeyError):
            ctx.get_op_context(999)

    def test_flatten_and_restore_round_trip(self):
        """Criterion 3: flatten → restore round-trips correctly."""
        ctx = NanoBatchContext(batch_idx=0)

        # Simulate forward: 3 ops saving tensors
        op0 = ctx.create_op_context(0)
        t0a, t0b = torch.tensor([1.0]), torch.tensor([2.0])
        op0.save_for_backward(t0a, t0b)

        op1 = ctx.create_op_context(1)
        t1a = torch.tensor([3.0])
        op1.save_for_backward(t1a)

        op2 = ctx.create_op_context(2)
        t2a, t2b, t2c = torch.tensor([4.0]), torch.tensor([5.0]), torch.tensor([6.0])
        op2.save_for_backward(t2a, t2b, t2c)

        # Flatten
        flat = ctx.flatten_saved_tensors()
        assert len(flat) == 6  # 2 + 1 + 3
        assert torch.equal(flat[0], t0a)
        assert torch.equal(flat[1], t0b)
        assert torch.equal(flat[2], t1a)
        assert torch.equal(flat[3], t2a)
        assert torch.equal(flat[4], t2b)
        assert torch.equal(flat[5], t2c)

        # to_save should be cleared
        assert op0.to_save is None
        assert op1.to_save is None
        assert op2.to_save is None

        # Restore from a tuple (as autograd provides)
        ctx.restore_saved_tensors(tuple(flat))

        assert op0.saved_tensors == (t0a, t0b)
        assert op1.saved_tensors == (t1a,)
        assert op2.saved_tensors == (t2a, t2b, t2c)

    def test_flatten_with_none_tensors(self):
        """Ops can save None tensors (e.g., optional bias)."""
        ctx = NanoBatchContext(batch_idx=0)

        op0 = ctx.create_op_context(0)
        t0 = torch.tensor([1.0])
        op0.save_for_backward(t0, None, t0)

        flat = ctx.flatten_saved_tensors()
        assert len(flat) == 3
        assert torch.equal(flat[0], t0)
        assert flat[1] is None
        assert torch.equal(flat[2], t0)

        ctx.restore_saved_tensors(tuple(flat))
        assert ctx.get_op_context(0).saved_tensors == (t0, None, t0)

    def test_flatten_with_no_saves(self):
        """Ops that don't call save_for_backward produce nothing."""
        ctx = NanoBatchContext(batch_idx=0)

        ctx.create_op_context(0)  # no save
        op1 = ctx.create_op_context(1)
        t1 = torch.tensor([42.0])
        op1.save_for_backward(t1)

        flat = ctx.flatten_saved_tensors()
        assert len(flat) == 1
        assert torch.equal(flat[0], t1)

        ctx.restore_saved_tensors(tuple(flat))
        assert ctx.get_op_context(1).saved_tensors == (t1,)
        # op 0 should not have saved_tensors set
        assert ctx.get_op_context(0).saved_tensors is None

    def test_flatten_preserves_op_id_order(self):
        """Tensors are flattened in op_id order, not insertion order."""
        ctx = NanoBatchContext(batch_idx=0)

        # Insert out of order
        op5 = ctx.create_op_context(5)
        op2 = ctx.create_op_context(2)

        t5 = torch.tensor([50.0])
        t2 = torch.tensor([20.0])
        op5.save_for_backward(t5)
        op2.save_for_backward(t2)

        flat = ctx.flatten_saved_tensors()
        # op_id 2 comes before op_id 5
        assert torch.equal(flat[0], t2)
        assert torch.equal(flat[1], t5)

        ctx.restore_saved_tensors(tuple(flat))
        assert ctx.get_op_context(2).saved_tensors == (t2,)
        assert ctx.get_op_context(5).saved_tensors == (t5,)

    def test_tensor_store_is_independent(self):
        """Each NanoBatchContext has its own TensorStore."""
        ctx0 = NanoBatchContext(batch_idx=0)
        ctx1 = NanoBatchContext(batch_idx=1)

        ctx0.tensor_store.set("t_0", torch.tensor([1.0]))
        assert ctx1.tensor_store.get("t_0") is None

    def test_batch_idx(self):
        ctx = NanoBatchContext(batch_idx=1)
        assert ctx.batch_idx == 1
