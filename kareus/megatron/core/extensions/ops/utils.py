"""Utility helpers for sub-context tensor management across fused operations."""

from transformer_engine.pytorch.ops.op import OperationContext


def merge_sub_contexts(
    ctx: OperationContext,
    sub_contexts: list[OperationContext],
) -> None:
    """Merge saved tensors from sub-contexts into a parent context for backward pass.

    Collects all tensors that sub-contexts want to save, stores them in the
    parent context via ``save_for_backward``, and records the index range so
    each sub-context's tensors can be restored later with
    :func:`restore_sub_contexts`.
    """
    to_save = []
    for sub_ctx in sub_contexts:
        if sub_ctx.to_save is not None:
            range_start = len(to_save)
            to_save.extend(sub_ctx.to_save)
            sub_ctx._saved_tensors_range = (range_start, len(to_save))
            sub_ctx.to_save = None
    if to_save:
        ctx.save_for_backward(*to_save)


def restore_sub_contexts(
    ctx: OperationContext,
    sub_contexts: list[OperationContext],
) -> None:
    """Restore saved tensors from a parent context back into sub-contexts.

    Reverses the packing done by :func:`merge_sub_contexts` so each
    sub-context sees only its own saved tensors during the backward pass.
    """
    if ctx.saved_tensors is None:
        return
    for sub_ctx in sub_contexts:
        if sub_ctx._saved_tensors_range is not None:
            sub_ctx.saved_tensors = ctx.saved_tensors[
                slice(*sub_ctx._saved_tensors_range)
            ]
