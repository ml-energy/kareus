"""
Merges ``BasicLinear`` and ``Bias`` into a single ``BasicOperation`` subclass
so that the PartitionFuser can call ``fuser_forward`` / ``fuser_backward``
as one atomic step.  This avoids the overhead of a ``FusedOperation`` pipeline
for the common linear+bias pattern.

The main motivation is to support the ``skip_bias_add=True`` pattern used by
Megatron-LM linear layers.  When``skip_bias_add=True``, the bias is not
fused into the GEMM but returned as a separate output so that it can be fused
with a later residual-add or dropout-add operation by the PartitionFuser, 
reducing kernel launch overhead. When ``apply_bias=True`` 
(i.e. ``skip_bias_add=False``), the bias is fused directly into the cuBLAS 
GEMM call instead.

Also serves as the base class for the composite ``Linear`` op.
"""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from transformer_engine.pytorch.ops._common import (
    canonicalize_device,
    canonicalize_dtype,
)

from .basic_linear import BasicLinear


class BasicLinearBias(BasicLinear):
    """Apply linear transformation with optional bias: :math:`y = x A^T + b`

    Merges BasicLinear and Bias into a single BasicOperation so that
    the partition system can call fuser_forward/fuser_backward directly.

    When ``return_bias=True`` (the ``skip_bias_add`` path from Megatron's
    ``TELinear`` and attention projections), the bias is returned as an
    extra output for downstream fusion with residual-add / dropout-add.
    When ``apply_bias=True``, the bias is fused into the cuBLAS GEMM.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
    has_bias: bool, default = True
        Whether to include a bias parameter
    apply_bias: bool, default = True
        Whether to fuse bias into the GEMM (cuBLAS bias fusion)
    return_bias: bool, default = False
        Whether to return bias as an extra output without fusing
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        has_bias: bool = True,
        apply_bias: bool = True,
        return_bias: bool = False,
        device=None,
        dtype=None,
        tensor_parallel_mode=None,
        tensor_parallel_group=None,
        tensor_parallel_size=None,
        sequence_parallel: bool = False,
        rng_state_tracker_function=None,
        accumulate_into_main_grad: bool = False,
        userbuffers_options=None,
        bias_fusable: bool = False,
        use_allreduce_buffer: tuple[bool, bool] = (False, False),
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            tensor_parallel_size=tensor_parallel_size,
            sequence_parallel=sequence_parallel,
            rng_state_tracker_function=rng_state_tracker_function,
            accumulate_into_main_grad=accumulate_into_main_grad,
            userbuffers_options=userbuffers_options,
            bias_fusable=bias_fusable,
            use_allreduce_buffer=use_allreduce_buffer,
        )

        self.has_bias: bool = has_bias
        self.apply_bias: bool = apply_bias
        self.return_bias: bool = return_bias

        if has_bias:
            device_obj = canonicalize_device(device)
            defer_param_init = device_obj.type == "meta"
            if defer_param_init:
                device_obj = canonicalize_device(None)

            bias_param = torch.empty(
                self.local_out_features,
                device="meta",
                dtype=canonicalize_dtype(dtype),
            )
            bias_param = torch.nn.Parameter(bias_param)
            self.bias: Optional[torch.nn.Parameter]
            self.register_parameter("bias", bias_param)
            if not defer_param_init:
                self.reset_bias()
        else:
            self.bias = None

        self.num_extra_outputs: int = 1 if (return_bias and has_bias) else 0

    def reset_bias(self) -> None:
        """Initialize bias parameter."""
        bias = self.bias
        if bias is None:
            return
        device = canonicalize_device(None)
        if bias.device.type != "cuda":
            bias = torch.empty_like(bias, device=device)
        else:
            bias = bias.to(device=device)
        bias.zero_()
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_forward(self, *args, **kwargs) -> None:
        super().pre_forward(*args, **kwargs)
        if self.has_bias and self.bias is not None and self.bias.device.type == "meta":
            self.reset_bias()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Check which grads are required
        input_requires_grad = ctx.requires_grad and input_.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight.requires_grad

        # FP8 metadata
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_quantizer = None
        weight_quantizer = None
        output_quantizer = None
        grad_output_quantizer = None
        grad_input_quantizer = None
        if with_quantized_compute:
            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            if next_op is not None and next_op.num_quantizers("forward") > 0:
                output_quantizer = next_op.get_quantizer("forward", 0)
            grad_output_quantizer = self.get_quantizer("backward", 0)
            if prev_op is not None and prev_op.num_quantizers("backward") > 0:
                grad_input_quantizer = prev_op.get_quantizer("backward", 0)
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            weight_quantizer.set_usage(rowwise=True, columnwise=False)

        # Get autocast dtype if needed
        dtype = None
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")

        # Persistent output for backward reuse (shared with AllReduce buffer)
        if self.use_allreduce_buffer_fwd and input_requires_grad:
            persist_out = self._get_persistent_output(batch_idx)
        else:
            persist_out = None

        # Linear forward with optional bias fusion into GEMM
        bias = self.bias if (self.apply_bias and self.has_bias) else None
        output, x_local, _ = BasicLinear._functional_forward(
            input=input_,
            weight=self.weight,
            bias=bias,
            out=persist_out,
            accumulate_into_out=False,
            dtype=dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
        )

        # Save state for backward pass
        ctx.save_for_backward(x_local)
        ctx.with_quantized_compute = with_quantized_compute
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.grad_input_quantizer = grad_input_quantizer
        ctx.dtype = dtype
        ctx.input_requires_grad = input_requires_grad
        ctx.weight_requires_grad = weight_requires_grad
        ctx.has_prev_op = prev_op is not None
        ctx.batch_idx = batch_idx

        # Return output with optional extra bias output
        if self.return_bias and self.has_bias:
            return output, self.bias
        return output, None

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
        grad_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:

        # Saved tensors from forward pass
        (x_local,) = ctx.saved_tensors

        # wgrad fusion
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weight = None
        if ctx.weight_requires_grad and accumulate_into_main_grad:
            raise NotImplementedError("Accumulate into main grad is not supported")
        else:
            accumulate_into_main_grad = False

        # Persistent output for backward reuse (shared with AllReduce buffer)
        if self.use_allreduce_buffer_bwd and ctx.input_requires_grad:
            persist_out = self._get_persistent_output(ctx.batch_idx)
        else:
            persist_out = None

        # Linear backward pass
        grad_input, grad_weight = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=self.weight,
            input_requires_grad=ctx.input_requires_grad,
            weight_requires_grad=ctx.weight_requires_grad,
            dtype=ctx.dtype,
            grad_input=persist_out,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=ctx.with_quantized_compute,
            input_quantizer=ctx.input_quantizer,
            weight_quantizer=ctx.weight_quantizer,
            grad_output_quantizer=ctx.grad_output_quantizer,
            grad_input_quantizer=ctx.grad_input_quantizer,
        )

        if accumulate_into_main_grad:
            grad_weight = None

        # Compute bias gradient
        if self.has_bias:
            if self.apply_bias:
                # Bias fused into GEMM: grad_bias = grad_output
                db = grad_output
            else:
                # return_bias mode: grad comes from upstream
                db = grad_bias
            return grad_input, [grad_weight, db]
        return grad_input, [grad_weight]

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, ...]]]:
        output, extra_out = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **basic_op_kwargs[0],
        )
        if extra_out is not None:
            return output, [(extra_out,)]
        return output, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[Iterable[Optional[torch.Tensor]]],
        list[tuple[()]],
    ]:
        grad_bias = None
        if basic_op_grad_extra_outputs and basic_op_grad_extra_outputs[0]:
            grad_bias, = basic_op_grad_extra_outputs[0]
        grad_input, grad_params = self.op_backward(
            basic_op_ctxs[0], grad_output, grad_bias,
        )
        return grad_input, [grad_params], [()]
