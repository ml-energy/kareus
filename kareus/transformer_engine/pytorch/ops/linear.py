"""
Modified from TransformerEngine (transformer_engine/pytorch/ops/linear.py).
Changes:
- Replaces the original ``FusedLinear`` (a ``FusedOperation`` pipeline of
  BasicLinear + Bias + AllReduce + optional activation) with a single
  ``BasicLinearBias`` subclass.  This flattens the operation graph so that
  the PartitionFuser can schedule each piece independently.
- Pre-canonicalizes tensor-parallel configuration (column/row splitting)
  and passes already-local dimensions to ``BasicLinearBias.__init__`` with
  ``tensor_parallel_mode=None``, because TP communication is handled
  externally by the fuser's AllReduce / AllGatherKV / ReduceScatterKV ops
  rather than inside the linear op itself.
- Removes Userbuffers-based TP overlap support (now handled by the fuser).
"""


from __future__ import annotations
from collections.abc import Callable
from typing import Optional, Union, Tuple

import torch

from transformer_engine.pytorch.distributed import CudaRNGStatesTracker

from kareus.transformer_engine.pytorch.ops.basic import (
    BasicLinear,
    BasicLinearBias,
)


class Linear(BasicLinearBias):
    """Modified from TransformerEngine's ``Linear``: inherits from
    ``BasicLinearBias`` instead of being a ``FusedLinear`` pipeline,
    enabling ``op_forward`` / ``op_backward`` to be called with an
    externally managed ``ctx`` by the PartitionFuser.  TP communication
    is factored out to separate AllReduce / AllGatherKV ops in the fuser.

    Apply linear transformation: :math:`y = x A^T + b`

    This is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
    bias: bool, default = `True`
        Apply additive bias
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
        Mode for tensor parallelism
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism
    sequence_parallel: bool, default = `False`
        Whether to apply sequence parallelism together with tensor
        parallelism, i.e. distributing input or output tensors along
        outer dimension (sequence or batch dim) when not distributing
        along inner dimension (embedding dim)
    rng_state_tracker_function: callable
        Function that returns CudaRNGStatesTracker, which is used for
        model-parallel weight initialization
    accumulate_into_main_grad: bool, default = `False`
        Whether to directly accumulate weight gradients into the
        weight's `main_grad` attribute instead of relying on PyTorch
        autograd. The weight's `main_grad` must be set externally and
        there is no guarantee that `grad` will be set or be
        meaningful.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        return_bias: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        accumulate_into_main_grad: bool = False,
        use_persistent_output: Union[bool, Tuple[bool, bool]] = False,
        num_batches: Optional[int] = 1,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> None:

        apply_bias = bias and not return_bias

        if tensor_parallel_mode == "column" and return_bias == False:
            bias_fusable = True
        else:
            bias_fusable = False

        # Canonicalize TP configuration before passing to super
        (
            tensor_parallel_mode,
            tensor_parallel_group,
            tensor_parallel_size,
            sequence_parallel,
            local_in_features,
            local_out_features,
        ) = BasicLinear._canonicalize_tensor_parallelism(
            mode=tensor_parallel_mode,
            process_group=tensor_parallel_group,
            tensor_parallel_size=tensor_parallel_size,
            sequence_parallel=sequence_parallel,
            in_features=in_features,
            out_features=out_features,
        )

        # Initialize BasicLinearBias with local (post-TP) dimensions
        # and tensor_parallel_mode=None since TP is already applied
        super().__init__(
            in_features=local_in_features,
            out_features=local_out_features,
            has_bias=bias,
            apply_bias=apply_bias,
            return_bias=return_bias,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=1,
            sequence_parallel=False,
            rng_state_tracker_function=rng_state_tracker_function,
            accumulate_into_main_grad=accumulate_into_main_grad,
            bias_fusable=bias_fusable,
            use_persistent_output=use_persistent_output,
            num_batches=num_batches,
            batch_size=batch_size,
            seq_length=seq_length,
        )

        if sequence_parallel:
            raise NotImplementedError("Sequence parallelism is not supported")
