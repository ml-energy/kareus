# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Manager class for a pipeline of fusible operations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Optional, Tuple

import torch

from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    FusibleOperation,
    OperationContext,
)
from kareus.transformer_engine.pytorch.ops.fused import (
    fuse_forward_linear_bias_activation,
)
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from kareus.megatron.core.extensions.ops import TEFusibleRowParallelLinear, TEFusibleColumnParallelLinear
from kareus.transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from kareus.transformer_engine.pytorch.ops.basic.bias import Bias
from kareus.megatron.core.extensions.ops import QKVPostProcessOp
from kareus.megatron.core.extensions.ops import RotaryEmbeddingOp
from kareus.transformer_engine.pytorch.attention.dot_product_attention import DotProductAttentionOp
from kareus.megatron.core.extensions.ops import BiasSwigluOp


class _PartitionFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
        func_ctx: Optional[torch.autograd.function.FunctionCtx],
        hidden_states: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        allreduce_input: Optional[torch.Tensor],
        allreduce_overlap_window: Optional[Tuple[int, int]],
        allreduce_sm_configs: Optional[Tuple[int, int]],
        allreduce_overlap_window_backward: Optional[Tuple[int, int]],
        allreduce_sm_configs_backward: Optional[Tuple[int, int]],
        allreduce_comm_op: Optional[FusibleOperation],
        forward_ops: list[tuple[FusibleOperation, list[int]]],
        backward_ops: list[tuple[FusibleOperation, list[int]]],
        basic_ops: list[BasicOperation],
        is_grad_enabled: bool,
        num_params: int,
        *params: torch.nn.Parameter,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass

        Parameters
        ----------
        func_ctx: torch.autograd.function.FunctionCtx
            Context for PyTorch autograd function
        input_: torch.Tensor
            Input to first operation in pipeline
        forward_ops: list of tuple
            Forward pass operations and the indices of the
            corresponding basic operations. The order should match
            basic_ops.
        backward_ops: list of tuple
            Backward pass operations and the indices of the
            corresponding basic operations. The order should be the
            reverse of basic_ops.
        basic_ops: list of BasicOperation
            Basic operations
        basic_op_kwargs: list of dict
            Keyword arguments to BasicOperation
        num_params: int
            Number of parameter tensors to include in autograd graph.
        *params_and_extra_inputs: torch.Tensor
            Other tensor inputs to include in autograd graph. Consists
            of parameter tensors, followed by extra operation inputs.

        Returns
        -------
        Output tensor(s). If none of the operations have any extra
        tensor outputs, then the pipeline's output tensor is returned.
        Otherwise, a tuple with the pipeline's output tensor and extra
        tensor outputs is returned.

        """

        # Operation autograd contexts
        basic_op_ctxs = [OperationContext() for _ in range(len(basic_ops))]

        current_stream = torch.cuda.current_stream()
        if allreduce_comm_op:
            comm_start, comm_end = allreduce_overlap_window
            if allreduce_sm_configs:
                sm_num, block_size = allreduce_sm_configs
            else:
                sm_num, block_size = None, None
        else:
            comm_start, comm_end = -1, -1

        if comm_start == -1 and allreduce_comm_op is not None:
            current_stream.synchronize()
            allreduce_comm_op.fuser_forward(
                [None], allreduce_input,
                basic_op_extra_inputs=[], basic_op_prev_ops=[None], basic_op_next_ops=[None], basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}]
            )
            allreduce_comm_op.sync()
        
        # if comm_start == 0:
        #     allreduce_comm_op.event_record(current_stream)

        # Apply forward ops
        x = hidden_states
        requires_grad = is_grad_enabled and x.requires_grad
        get_residual = False
        get_bias = False

        for fused_idx, (op, basic_op_idxs) in enumerate(forward_ops):

            # Get extra inputs
            extra_inputs = []
            if isinstance(op, BiasDropoutAddOp):
                extra_inputs = [(bias, residual)]
            elif isinstance(op, LayerNorm) or isinstance(op, RMSNorm):
                residual = x
                get_residual = True
            elif isinstance(op, RotaryEmbeddingOp):
                extra_inputs = [(key, rotary_pos_emb)]
            elif isinstance(op, DotProductAttentionOp):
                extra_inputs = [(key, value)]
            elif isinstance(op, BiasSwigluOp):
                assert get_bias == True
                get_bias = False
                extra_inputs = [(bias,)]

            # Check if backward op is required
            if is_grad_enabled:
                if not requires_grad:
                    requires_grad = any(param.requires_grad for param in op.parameters())
                if not requires_grad:
                    requires_grad = any(any(x.requires_grad for x in xs) for xs in extra_inputs)
            for idx in basic_op_idxs:
                basic_op_ctxs[idx].requires_grad = requires_grad
            if requires_grad != x.requires_grad:
                if requires_grad:
                    x.requires_grad_()
                else:
                    x = x.detach()

            # Forward op
            prev_ops = [basic_ops[idx - 1] if idx > 0 else None for idx in basic_op_idxs]
            next_ops = [
                basic_ops[idx + 1] if (idx < len(basic_ops) - 1) else None for idx in basic_op_idxs
            ]

            if comm_start == fused_idx:
                # Wait for the event from the previous operation before starting allreduce
                # allreduce_comm_op.event_wait()
                current_stream.synchronize()
                allreduce_comm_op.fuser_forward(
                    [OperationContext()],
                    allreduce_input,
                    basic_op_extra_inputs=[], basic_op_prev_ops=[None], basic_op_next_ops=[None], basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}]
                )

            x, fused_op_extra_outputs = op.fuser_forward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                x,
                basic_op_extra_inputs=extra_inputs,
                basic_op_prev_ops=prev_ops,
                basic_op_next_ops=next_ops,
                basic_op_kwargs=[{} for _ in basic_op_idxs],
            )

            # Record event after the operation at fused_idx-1 completes
            # if fused_idx == comm_start - 1:
            #     allreduce_comm_op.event_record(current_stream)

            if comm_end == fused_idx:
                allreduce_comm_op.sync()

            # Get extra outputs
            if isinstance(op, QKVPostProcessOp):
                key, value = fused_op_extra_outputs[0]
            elif isinstance(op, RotaryEmbeddingOp):
                key = fused_op_extra_outputs[0][0]
            elif isinstance(op, Bias) and op.return_bias:
                bias = fused_op_extra_outputs[0][0]
                get_bias = True

            x.requires_grad_(requires_grad=requires_grad)
            for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
                for y in ys:
                    if y is not None:
                        y.requires_grad_(requires_grad=requires_grad)

        # Save context for backward pass
        if is_grad_enabled:

            # Flatten list of saved tensors
            to_save = []
            for ctx in basic_op_ctxs:
                range_start = len(to_save)
                if ctx.to_save is not None:
                    to_save.extend(ctx.to_save)
                range_end = len(to_save)
                ctx.to_save = None
                ctx._saved_tensors_range = (range_start, range_end)
            func_ctx.save_for_backward(*to_save)

            # Other context
            func_ctx.allreduce_window_backward = allreduce_overlap_window_backward
            func_ctx.allreduce_sm_configs_backward = allreduce_sm_configs_backward
            func_ctx.allreduce_comm_op = allreduce_comm_op
            func_ctx.backward_ops = backward_ops
            func_ctx.basic_ops = basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = [sum(1 for _ in op.parameters()) for op in basic_ops]

        current_stream.synchronize()
        assert get_bias and get_residual, f"get_bias: {get_bias}, get_residual: {get_residual}"
        return x, bias, residual, allreduce_input

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_output: torch.Tensor,
        grad_bias: torch.Tensor,
        grad_residual: torch.Tensor,
        grad_allreduce_input: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        allreduce_comm_op = func_ctx.allreduce_comm_op
        allreduce_overlap_window = func_ctx.allreduce_window_backward
        allreduce_sm_configs = func_ctx.allreduce_sm_configs_backward
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Unflatten list of saved tensors
        for ctx in basic_op_ctxs:
            if ctx._saved_tensors_range is not None:
                ctx.saved_tensors = func_ctx.saved_tensors[slice(*ctx._saved_tensors_range)]

        current_stream = torch.cuda.current_stream()
        if allreduce_comm_op:
            comm_start, comm_end = allreduce_overlap_window
            if allreduce_sm_configs:
                sm_num, block_size = allreduce_sm_configs
            else:
                sm_num, block_size = None, None
        else:
            comm_start, comm_end = -1, -1

        if comm_start == -1 and allreduce_comm_op is not None:
            current_stream.synchronize()
            allreduce_comm_op.fuser_forward(
                [None], grad_allreduce_input,
                basic_op_extra_inputs=[], basic_op_prev_ops=[None], basic_op_next_ops=[None], basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}]
            )
            allreduce_comm_op.sync()
    
        # if comm_start == 0:
        #     allreduce_comm_op.event_record(current_stream)

        # Apply backward ops
        dx = grad_output
        grad_params = [None for _ in range(len(basic_ops))]
        get_grad_bias = False
        get_grad_residual = False
        grad_rotary_pos_emb = None
        
        for fused_idx, (op, basic_op_idxs) in enumerate(backward_ops):

            # Stop if no more gradients are required
            if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
                dx = None
                break

            # Get extra input gradients based on operation type
            grad_extra_outputs = [()]
            if isinstance(op, QKVPostProcessOp):
                grad_extra_outputs = [(grad_key, grad_value)]
            elif isinstance(op, RotaryEmbeddingOp):
                grad_extra_outputs = [(grad_key,)]
            elif isinstance(op, Bias) and op.return_bias:
                grad_extra_outputs = [(grad_bias,)]

            if comm_start == fused_idx:
                # allreduce_comm_op.event_wait()
                current_stream.synchronize()
                allreduce_comm_op.fuser_forward(
                    [None], grad_allreduce_input,
                    basic_op_extra_inputs=[], basic_op_prev_ops=[None], basic_op_next_ops=[None], basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}]
                )

            # Backward op
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
                [basic_op_ctxs[idx] for idx in basic_op_idxs],
                dx,
                basic_op_grad_extra_outputs=grad_extra_outputs,
            )
            for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
                grad_params[idx] = dparams
                basic_op_ctxs[idx].saved_tensors = None

            # if fused_idx == comm_start - 1:
            #     allreduce_comm_op.event_record(current_stream)

            if comm_end == fused_idx:
                allreduce_comm_op.sync()

            if isinstance(op, BiasDropoutAddOp):
                grad_bias, grad_residual = fused_op_grad_extra_inputs[0]
                get_grad_bias = True
                get_grad_residual = True
            elif isinstance(op, LayerNorm) or isinstance(op, RMSNorm):
                assert get_grad_residual == False
                dx = dx + grad_residual
                grad_residual = None
            elif isinstance(op, RotaryEmbeddingOp):
                grad_key, grad_rotary_pos_emb = fused_op_grad_extra_inputs[0]
            elif isinstance(op, DotProductAttentionOp):
                grad_key, grad_value = fused_op_grad_extra_inputs[0]
            elif isinstance(op, BiasSwigluOp):
                grad_bias = fused_op_grad_extra_inputs[0][0]
            elif isinstance(op, Bias) and op.return_bias:
                grad_bias = None

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, dparams in enumerate(grad_params):
            num_params = func_ctx.basic_op_num_params[idx]
            if dparams is None:
                dparams = [None for _ in range(num_params)]
            else:
                dparams = list(dparams)
            if len(dparams) != num_params:
                raise RuntimeError(
                    f"Expected op {idx} to generate {num_params} param grads, "
                    f"but got {len(dparams)}"
                )
            grad_params_flat.extend(dparams)

        current_stream.synchronize()
        return (
            dx,  # hidden_states
            grad_bias,  # bias
            grad_residual,  # residual  
            grad_rotary_pos_emb,  # rotary_pos_emb
            None,  # attention_mask
            grad_allreduce_input,  # allreduce_input
            None,  # allreduce_overlap_window
            None,  # allreduce_sm_configs
            None,  # allreduce_overlap_window_backward
            None,  # allreduce_sm_configs_backward
            None,  # allreduce_comm_op
            None,  # forward_ops
            None,  # backward_ops
            None,  # basic_ops
            None,  # is_grad_enabled
            None,  # num_params
            *grad_params_flat,
        )


class PartitionFuser:
    """Manages forward and backward passes for a pipeline of operations

    Parameters
    ----------
    ops: list of FusibleOperation
        Pipeline of operations
    fuse_ops: bool, default = `True`
        Whether to attempt fusing operations

    """

    def __init__(
        self,
        ops: list[FusibleOperation],
        allreduce_comm_op: Optional[FusibleOperation] = None,
        fuse_ops: bool = True,    
    ) -> None:

        # Get list of basic operations
        basic_ops = []
        for op in ops:
            if op.is_fused_op:
                basic_ops.extend(op.basic_ops)
            else:
                basic_ops.append(op)
        self._num_basic_ops: int = len(basic_ops)
        self._basic_ops: list[BasicOperation] = basic_ops

        # Number of extra tensor inputs
        self._num_extra_inputs: int = sum(op.num_extra_inputs for op in basic_ops)

        # Ops for forward and backward pass
        self._forward_ops: list[tuple[FusibleOperation, list[int]]]
        self._backward_ops: list[tuple[FusibleOperation, list[int]]]
        self._forward_ops = [(op, (idx,)) for idx, op in enumerate(self._basic_ops)]
        self._backward_ops = list(reversed(self._forward_ops))

        self._allreduce_comm_op = allreduce_comm_op

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()
        
        self.num_forward_ops = len(self._forward_ops)
        self.num_backward_ops = len(self._backward_ops)

    @classmethod
    def _fuse_forward_ops(
        cls,
        ops: list[tuple[FusibleOperation, list[int]]],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Attempt to fuse operations in forward pass"""
        # ops = fuse_userbuffers_forward_linear(ops)
        # ops = fuse_forward_linear_bias_add(ops)
        ops = fuse_forward_linear_bias_activation(ops)
        return ops

    @classmethod
    def _fuse_backward_ops(
        cls,
        ops: list[tuple[FusibleOperation, list[int]]],
    ) -> list[tuple[FusibleOperation, list[int]]]:
        """Attempt to fuse operations in backward pass"""
        # ops = fuse_userbuffers_backward_linear(ops)
        # ops = fuse_backward_linear_add(ops)
        return ops

    def fuse_ops(self) -> None:
        """Attempt to fuse operations"""
        self._forward_ops = self._fuse_forward_ops(self._forward_ops)
        # self._backward_ops = self._fuse_backward_ops(self._backward_ops)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        residual: torch.Tensor = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        allreduce_input: Optional[torch.Tensor] = None,
        allreduce_overlap_window: Optional[Tuple[int, int]] = None,
        allreduce_sm_configs: Optional[Tuple[int, int]] = None,
        allreduce_overlap_window_backward: Optional[Tuple[int, int]] = None,
        allreduce_sm_configs_backward: Optional[Tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, ...]: # hidden_states, bias, residual

        # Initialization before forward pass
        for op in self._basic_ops:
            op.pre_forward()

        # Flatten list of parameters
        params = [param for op in self._basic_ops for param in op.parameters()]

        # Fuser forward pass
        is_grad_enabled = torch.is_grad_enabled()
        if is_grad_enabled:
            forward_func = _PartitionFuserAutogradFunction.apply
            args = []
        else:
            forward_func = _PartitionFuserAutogradFunction.forward
            args = [None]
        args += (
            hidden_states,
            bias,
            residual,
            rotary_pos_emb,
            attention_mask,
            allreduce_input,
            allreduce_overlap_window,
            allreduce_sm_configs,
            allreduce_overlap_window_backward,
            allreduce_sm_configs_backward,
            self._allreduce_comm_op,
            self._forward_ops,
            self._backward_ops,
            self._basic_ops,
            is_grad_enabled,
            len(params),
            *params,
        )
        return forward_func(*args)