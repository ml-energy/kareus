"""Runtime context management for partition-based execution.

Provides :class:`TensorStore` for tensor routing by auto-generated IDs
and :class:`NanoBatchContext` for per-nanobatch state (tensor storage +
per-operator saved activations for backward).

Cross-nanobatch pattern:
    ctx      = NanoBatchContext for THIS nanobatch
    pre_ctx  = NanoBatchContext for the OTHER nanobatch

    Compute ops: read/write via ctx.tensor_store
    Comm ops:    read/write via pre_ctx.tensor_store
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .tensor_graph import TensorPort

from transformer_engine.pytorch.ops.op import OperationContext


# ------------------------------------------------------------------ #
#  TensorStore
# ------------------------------------------------------------------ #


@dataclass
class TensorStore:
    """Dict-based storage mapping tensor IDs to tensors.

    Tensor IDs (``t_0``, ``t_1``, ``ext_main``, etc.) are assigned at
    graph-build time by :class:`TensorGraphBuilder` and used at runtime
    for automatic tensor routing between operators.
    """

    _tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def set(self, tensor_id: str, tensor: torch.Tensor) -> None:
        """Store *tensor* under *tensor_id*."""
        self._tensors[tensor_id] = tensor

    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Retrieve the tensor for *tensor_id*, or ``None`` if absent."""
        return self._tensors.get(tensor_id)

    def get_by_ports(self, ports: List[TensorPort]) -> List[Optional[torch.Tensor]]:
        """Read tensors for a list of :class:`TensorPort` in port order."""
        return [self._tensors.get(p.tensor_id) for p in ports]

    def set_from_ports(
        self,
        ports: List[TensorPort],
        tensors: List[Optional[torch.Tensor]],
    ) -> None:
        """Write tensors for a list of :class:`TensorPort`, skipping ``None``."""
        for port, tensor in zip(ports, tensors):
            if tensor is not None:
                self._tensors[port.tensor_id] = tensor


# ------------------------------------------------------------------ #
#  NanoBatchContext
# ------------------------------------------------------------------ #


@dataclass
class NanoBatchContext:
    """Context for a single nano-batch across all partitions.

    Holds:
    - ``tensor_store``: tensors stored by auto-generated IDs (t_0, t_1, ...)
    - ``op_contexts``: per-operator :class:`OperationContext` for save/restore

    Lifecycle:
    1. **Forward**: each compute op calls ``create_op_context(op_id)``
       to get an ``OperationContext`` for saving activations.
    2. **Between forward and backward**: ``flatten_saved_tensors()``
       collects all saved tensors into a flat list for
       ``torch.autograd.Function.save_for_backward()``.
    3. **Backward start**: ``restore_saved_tensors(saved_tensors)``
       slices the flat tuple back into each ``OperationContext``.
    4. **Backward**: each compute op calls ``get_op_context(op_id)``
       to retrieve the context with restored ``saved_tensors``.
    """

    batch_idx: int  # 0 or 1

    # Tensor storage — tensors stored by auto-generated IDs
    tensor_store: TensorStore = field(default_factory=TensorStore)

    # Operation contexts for backward, keyed by op_id (ComputeOp.op_id)
    op_contexts: Dict[int, OperationContext] = field(default_factory=dict)

    # Bookkeeping for save_for_backward flattening
    _saved_tensors: List[torch.Tensor] = field(default_factory=list)
    _saved_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def create_op_context(self, op_id: int) -> OperationContext:
        """Create and register an :class:`OperationContext` during forward.

        Args:
            op_id: Unique ID from ``ComputeOp.op_id``, assigned by
                :class:`TensorGraphBuilder`.

        Returns:
            A fresh ``OperationContext`` stored in ``self.op_contexts[op_id]``.
        """
        ctx = OperationContext()
        self.op_contexts[op_id] = ctx
        return ctx

    def get_op_context(self, op_id: int) -> OperationContext:
        """Retrieve the :class:`OperationContext` saved during forward.

        Args:
            op_id: Same ``op_id`` used in ``create_op_context``.

        Returns:
            The ``OperationContext`` with ``saved_tensors`` restored
            (after ``restore_saved_tensors`` has been called).
        """
        return self.op_contexts[op_id]

    def flatten_saved_tensors(self) -> List[Optional[torch.Tensor]]:
        """Flatten all ``OperationContext.to_save`` into a single list.

        Called after all forward partitions execute, before
        ``save_for_backward()``.

        Iterates ``op_contexts`` in ``op_id`` order.  For each context
        with tensors to save, records ``(range_start, range_end)`` in
        ``_saved_ranges`` and clears ``ctx.to_save``.

        Returns:
            Flat list of tensors for ``save_for_backward()``.
        """
        self._saved_tensors = []
        self._saved_ranges = {}

        for op_id in sorted(self.op_contexts):
            ctx = self.op_contexts[op_id]
            if ctx.to_save is not None and len(ctx.to_save) > 0:
                start = len(self._saved_tensors)
                self._saved_tensors.extend(ctx.to_save)
                end = len(self._saved_tensors)
                self._saved_ranges[op_id] = (start, end)
                ctx.to_save = None

        return self._saved_tensors

    def restore_saved_tensors(
        self,
        saved_tensors: Tuple[Optional[torch.Tensor], ...],
    ) -> None:
        """Restore saved tensors back into each :class:`OperationContext`.

        Called at the start of backward.  Uses ``_saved_ranges`` recorded
        during ``flatten_saved_tensors`` to slice the flat tuple.

        Args:
            saved_tensors: The tuple from ``ctx.saved_tensors`` in the
                autograd function's backward.
        """
        for op_id, (start, end) in self._saved_ranges.items():
            self.op_contexts[op_id].saved_tensors = saved_tensors[start:end]
