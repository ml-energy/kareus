"""Unified autograd boundary for graph-based transformer block execution.

Provides :class:`TransformerBlockAutogradFunction` — a single
``torch.autograd.Function`` wrapping the entire transformer block
(all layers, all partitions, both nanobatches).

Replaces the per-fuser autograd functions:
  - ``_PartitionFuserAutogradFunction``
  - ``_QKVFuserAutogradFunction``
  - ``_AttnOprojFuserAutogradFunction``

All tensor routing is automatic via :class:`TensorStore` + tensor IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .context_manager import NanoBatchContext, TensorStore
from .tensor_graph import TensorGraph


# ------------------------------------------------------------------ #
#  Seed configuration
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class SeedConfig:
    """Maps named tensor arguments to graph-level tensor IDs.

    Created by ``transformer_block`` at graph-build time and passed
    into :class:`TransformerBlockAutogradFunction` so that forward and
    backward can seed / read :class:`TensorStore` entries correctly.

    The tensor IDs here must match those used in
    ``TensorGraphBuilder.add_initial_channels`` when constructing the
    forward and backward graphs.
    """

    # Forward: tensor_id for the main per-nanobatch input (h1 / h2).
    h_tid: str = "ext_main"

    # Forward: tensor_id for rotary positional embedding (shared).
    # Set to ``None`` if the model doesn't use rotary embeddings.
    rotary_tid: Optional[str] = "ext_rotary_pos_emb"

    # Forward: tensor_id for attention mask (shared).
    # Set to ``None`` if not routed through the graph.
    mask_tid: Optional[str] = None

    # Forward: channel name to read the final output from
    # ``forward_tensor_graph.channel_registry``.
    fwd_output_channel: str = "main"

    # Backward: tensor_id for seeding grad_h1 / grad_h2.
    bwd_grad_tid: str = "ext_grad_main"

    # Backward: channel name to read the final input gradient from
    # ``backward_tensor_graph.channel_registry``.
    bwd_output_channel: str = "grad_main"


# ------------------------------------------------------------------ #
#  Gradient combination helper
# ------------------------------------------------------------------ #


def _combine_param_grads(
    grad_params_nb1: Dict[int, List],
    grad_params_nb2: Dict[int, List],
    num_params: int,
) -> List[Optional[Tensor]]:
    """Sum parameter gradients from both nanobatches.

    Iterates ``op_id`` values in sorted order (matching the parameter
    collection order used by the transformer block).  For each
    ``op_id``, element-wise sums ``g_nb0 + g_nb1`` in-place.

    Args:
        grad_params_nb1: ``{op_id: [grad, ...]}`` accumulated from
            all NB0 backward partitions.
        grad_params_nb2: ``{op_id: [grad, ...]}`` accumulated from
            all NB1 backward partitions.
        num_params: Expected total count (must match ``len(*params)``
            from forward).

    Returns:
        Flat list of combined gradients in ``*params`` order.
    """
    combined: List[Optional[Tensor]] = []
    all_op_ids = sorted(
        set(grad_params_nb1.keys()) | set(grad_params_nb2.keys())
    )

    for op_id in all_op_ids:
        grads_1 = grad_params_nb1.get(op_id, [])
        grads_2 = grad_params_nb2.get(op_id, [])
        max_len = max(len(grads_1), len(grads_2))
        assert len(grads_1) == len(grads_2), (
            f"Gradient length mismatch: {len(grads_1)} != {len(grads_2)}"
        )

        for i in range(max_len):
            g1 = grads_1[i] if i < len(grads_1) else None
            g2 = grads_2[i] if i < len(grads_2) else None

            if g1 is not None and g2 is not None:
                combined.append(g1.add_(g2))
            elif g1 is not None:
                combined.append(g1)
            elif g2 is not None:
                combined.append(g2)
            else:
                combined.append(None)

    assert len(combined) == num_params, (
        f"Parameter gradient count mismatch: expected {num_params}, "
        f"got {len(combined)}"
    )

    return combined


# ------------------------------------------------------------------ #
#  Unified autograd function
# ------------------------------------------------------------------ #


class TransformerBlockAutogradFunction(torch.autograd.Function):
    """Single autograd boundary wrapping the entire transformer block.

    All layers, partitions, and both nanobatches are enclosed in one
    ``torch.autograd.Function``.  Tensor routing is fully automatic
    via :class:`TensorStore` + tensor IDs.

    Forward signature::

        forward(func_ctx, h1, h2, rotary_pos_emb, attention_mask,
                forward_partitions, backward_partitions,
                forward_tensor_graph, backward_tensor_graph,
                scheduler, config, seed_config, *params)

    Backward return (matches forward arg order)::

        (dh1, dh2, None, None,     # rotary_pos_emb, attention_mask
         None, None,               # forward_partitions, backward_partitions
         None, None,               # forward_tensor_graph, backward_tensor_graph
         None, None, None,         # scheduler, config, seed_config
         *combined_grad_params)
    """

    @staticmethod
    def forward(
        func_ctx,
        h1: Tensor,
        h2: Tensor,
        rotary_pos_emb: Optional[Tensor],
        attention_mask: Optional[Tensor],
        forward_partitions,       # List[ForwardPartition]
        backward_partitions,      # List[BackwardPartition]
        forward_tensor_graph: TensorGraph,
        backward_tensor_graph: TensorGraph,
        scheduler,                # PipelineCommScheduler | None
        config,                   # TransformerConfig
        seed_config: SeedConfig,
        *params: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # -- 1. Create NanoBatchContexts --
        ctx_nb1 = NanoBatchContext(batch_idx=0)
        ctx_nb2 = NanoBatchContext(batch_idx=1)

        # -- 2. Seed TensorStores with initial tensors --
        ctx_nb1.tensor_store.set(seed_config.h_tid, h1)
        ctx_nb2.tensor_store.set(seed_config.h_tid, h2)

        if rotary_pos_emb is not None and seed_config.rotary_tid is not None:
            ctx_nb1.tensor_store.set(seed_config.rotary_tid, rotary_pos_emb)
            ctx_nb2.tensor_store.set(seed_config.rotary_tid, rotary_pos_emb)

        if attention_mask is not None and seed_config.mask_tid is not None:
            ctx_nb1.tensor_store.set(seed_config.mask_tid, attention_mask)
            ctx_nb2.tensor_store.set(seed_config.mask_tid, attention_mask)

        # -- 3. Load schedule and execute forward partitions --
        current_schedule = (
            scheduler.current_schedule if scheduler is not None else None
        )

        for partition in forward_partitions:
            if current_schedule is not None:
                partition.load_schedule(current_schedule)

            if partition.nano_batch_idx == 0:
                partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
            else:
                partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)

        # -- 4. Extract final outputs --
        final_tid = forward_tensor_graph.get_output_channel(
            seed_config.fwd_output_channel
        )
        h1_out = ctx_nb1.tensor_store.get(final_tid)
        h2_out = ctx_nb2.tensor_store.get(final_tid)

        # -- 5. Save for backward --
        func_ctx.backward_partitions = backward_partitions
        func_ctx.backward_tensor_graph = backward_tensor_graph
        func_ctx.scheduler = scheduler
        func_ctx.seed_config = seed_config
        func_ctx.ctx_nb1 = ctx_nb1
        func_ctx.ctx_nb2 = ctx_nb2
        func_ctx.num_params = len(params)

        saved_1 = ctx_nb1.flatten_saved_tensors()
        saved_2 = ctx_nb2.flatten_saved_tensors()
        func_ctx.num_saved_1 = len(saved_1)
        func_ctx.save_for_backward(*saved_1, *saved_2)

        return h1_out, h2_out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx,
        grad_h1: Tensor,
        grad_h2: Tensor,
    ) -> tuple:

        # -- 1. Restore saved tensors --
        saved = func_ctx.saved_tensors
        num_saved_1: int = func_ctx.num_saved_1
        ctx_nb1: NanoBatchContext = func_ctx.ctx_nb1
        ctx_nb2: NanoBatchContext = func_ctx.ctx_nb2

        ctx_nb1.restore_saved_tensors(saved[:num_saved_1])
        ctx_nb2.restore_saved_tensors(saved[num_saved_1:])

        seed_config: SeedConfig = func_ctx.seed_config

        # -- 2. Seed backward TensorStores with grad inputs --
        # Use fresh TensorStores for backward grad routing.  Forward
        # activations are accessed via OperationContext.saved_tensors
        # (restored above), not via TensorStore.
        ctx_nb1.tensor_store = TensorStore()
        ctx_nb2.tensor_store = TensorStore()

        ctx_nb1.tensor_store.set(seed_config.bwd_grad_tid, grad_h1)
        ctx_nb2.tensor_store.set(seed_config.bwd_grad_tid, grad_h2)

        # -- 3. Load schedule and execute backward partitions --
        current_schedule = (
            func_ctx.scheduler.current_schedule
            if func_ctx.scheduler is not None
            else None
        )

        all_grad_params_nb1: Dict[int, List] = {}
        all_grad_params_nb2: Dict[int, List] = {}

        for partition in func_ctx.backward_partitions:
            if current_schedule is not None:
                partition.load_schedule(current_schedule)

            if partition.nano_batch_idx == 0:
                grad_params = partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
                all_grad_params_nb1.update(grad_params)
            else:
                grad_params = partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
                all_grad_params_nb2.update(grad_params)

        # -- 4. Extract final input gradients --
        final_grad_tid = func_ctx.backward_tensor_graph.get_output_channel(
            seed_config.bwd_output_channel
        )
        dh1 = (
            ctx_nb1.tensor_store.get(final_grad_tid)
            if final_grad_tid is not None
            else None
        )
        dh2 = (
            ctx_nb2.tensor_store.get(final_grad_tid)
            if final_grad_tid is not None
            else None
        )

        # -- 5. Combine parameter gradients from both nanobatches --
        combined = _combine_param_grads(
            all_grad_params_nb1,
            all_grad_params_nb2,
            func_ctx.num_params,
        )

        # -- 6. Return gradients --
        # Must match forward signature:
        #   h1, h2, rotary_pos_emb, attention_mask,
        #   forward_partitions, backward_partitions,
        #   forward_tensor_graph, backward_tensor_graph,
        #   scheduler, config, seed_config, *params
        return (
            dh1,     # h1
            dh2,     # h2
            None,    # rotary_pos_emb
            None,    # attention_mask
            None,    # forward_partitions
            None,    # backward_partitions
            None,    # forward_tensor_graph
            None,    # backward_tensor_graph
            None,    # scheduler
            None,    # config
            None,    # seed_config
            *combined,
        )
