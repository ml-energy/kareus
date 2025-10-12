# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Manager class for a pipeline of fusible operations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Optional, Tuple, List

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
from kareus.megatron.core.extensions.ops import BiasGeluOp
from kareus.megatron.core.extensions.ops import BiasGegluOp

WAIT_EVENT = torch.cuda.Event()


def run_partition_ao_ag(
    basic_op_ctxs,
    query,
    comm_key,
    comm_value,
    forward_ops, 
    comm_op_fwd, 
    comm_overlap_window, 
    comm_sm_configs,
    is_grad_enabled,
    profile_ao_ag,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_fwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_fwd is not None:
        # current_stream.synchronize()
        comm_op_fwd.fuser_forward(
            [None], comm_key,
            basic_op_extra_inputs=[(comm_value,)], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size
            }]
        )
        # comm_op_fwd.sync()
        WAIT_EVENT.record(comm_op_fwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)
    
    # if not profile:
    #     if comm_start == 0:
    #         comm_op_fwd.event_record(current_stream)

    # Apply forward ops
    x = query
    requires_grad = is_grad_enabled and x.requires_grad
    get_residual = False
    get_bias = False

    for fused_idx, (op, basic_op_idxs) in enumerate(forward_ops):

        # Get extra inputs
        extra_inputs = []
        kwargs = [{} for _ in basic_op_idxs]
        # if isinstance(op, BiasDropoutAddOp):
        #     extra_inputs = [(bias, residual)]
        # elif isinstance(op, LayerNorm) or isinstance(op, RMSNorm):
        #     residual = x
        #     get_residual = True
        # elif isinstance(op, RotaryEmbeddingOp):
        #     extra_inputs = [(key, rotary_pos_emb)]
        if isinstance(op, DotProductAttentionOp):
            extra_inputs = [(None, None)] # Read from global variables
            kwargs[0]['batch_idx'] = 0
        # elif isinstance(op, BiasSwigluOp) or isinstance(op, BiasGeluOp) or isinstance(op, BiasGegluOp):
        #     assert get_bias == True
        #     get_bias = False
        #     extra_inputs = [(bias,)]

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

        if comm_start == fused_idx:
            # # Wait for the event from the previous operation before starting allreduce
            # if not profile:
            #     comm_op_fwd.event_wait()
            # else:
            #     current_stream.synchronize()
            # Synchronization in all_gather_kv
            comm_op_fwd.fuser_forward(
                [None], comm_key,
                basic_op_extra_inputs=[(comm_value,)], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size
                }]
            )

        x, fused_op_extra_outputs = op.fuser_forward(
            [basic_op_ctxs[idx] for idx in basic_op_idxs],
            x,
            basic_op_extra_inputs=extra_inputs,
            basic_op_prev_ops=[None],
            basic_op_next_ops=[None],
            basic_op_kwargs=kwargs,
        )

        # if not profile:
        #     # Record event after the operation at fused_idx-1 completes
        #     if fused_idx == comm_start - 1:
        #         comm_op_fwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_fwd.sync()

        # Get extra outputs
        # if isinstance(op, QKVPostProcessOp):
        #     key, value = fused_op_extra_outputs[0]
        # elif isinstance(op, RotaryEmbeddingOp):
        #     key = fused_op_extra_outputs[0][0]
        if isinstance(op, Bias) and op.return_bias:
            bias = fused_op_extra_outputs[0][0]
            get_bias = True

        x.requires_grad_(requires_grad=requires_grad)
        for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
            for y in ys:
                if y is not None:
                    y.requires_grad_(requires_grad=requires_grad)
    
    # if profile_ao_ag:
    #     current_stream.synchronize()
    # if comm_op_fwd is not None:
    #     comm_op_fwd.sync()
    WAIT_EVENT.record(comm_op_fwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)

    assert get_bias == True
    return x, bias


def run_partition_ao_ar(
    basic_op_ctxs,
    query,
    comm_input,
    forward_ops, 
    comm_op_fwd, 
    comm_overlap_window, 
    comm_sm_configs,
    is_grad_enabled,
    profile_ao_ag,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_fwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_fwd is not None:
        # current_stream.synchronize()
        comm_op_fwd.event_record(current_stream)
        comm_op_fwd.event_wait()
        comm_op_fwd.fuser_forward(
            [None], comm_input,
            basic_op_extra_inputs=[], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size
            }]
        )
        # comm_op_fwd.sync()
        WAIT_EVENT.record(comm_op_fwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)
    
    # if not profile_ao_ag:
    if comm_start == 0:
        comm_op_fwd.event_record(current_stream)

    # Apply forward ops
    x = query
    requires_grad = is_grad_enabled and x.requires_grad
    get_residual = False
    get_bias = False

    for fused_idx, (op, basic_op_idxs) in enumerate(forward_ops):

        # Get extra inputs
        extra_inputs = []
        kwargs = [{} for _ in basic_op_idxs]
        # if isinstance(op, BiasDropoutAddOp):
        #     extra_inputs = [(bias, residual)]
        # elif isinstance(op, LayerNorm) or isinstance(op, RMSNorm):
        #     residual = x
        #     get_residual = True
        # elif isinstance(op, RotaryEmbeddingOp):
        #     extra_inputs = [(key, rotary_pos_emb)]
        if isinstance(op, DotProductAttentionOp):
            extra_inputs = [(None, None)] # Read from global variables
            kwargs[0]['batch_idx'] = 1
        # elif isinstance(op, BiasSwigluOp) or isinstance(op, BiasGeluOp) or isinstance(op, BiasGegluOp):
        #     assert get_bias == True
        #     get_bias = False
        #     extra_inputs = [(bias,)]

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

        if comm_start == fused_idx:
            # # Wait for the event from the previous operation before starting allreduce
            # if not profile_ao_ag:
            comm_op_fwd.event_wait()
            # else:
            #     current_stream.synchronize()
            comm_op_fwd.fuser_forward(
                [None], comm_input,
                basic_op_extra_inputs=[], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size
                }]
            )

        x, fused_op_extra_outputs = op.fuser_forward(
            [basic_op_ctxs[idx] for idx in basic_op_idxs],
            x,
            basic_op_extra_inputs=extra_inputs,
            basic_op_prev_ops=[None],
            basic_op_next_ops=[None],
            basic_op_kwargs=kwargs,
        )

        # if not profile_ao_ag:
        # Record event after the operation at fused_idx-1 completes
        if fused_idx == comm_start - 1:
            comm_op_fwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_fwd.sync()

        # Get extra outputs
        # if isinstance(op, QKVPostProcessOp):
        #     key, value = fused_op_extra_outputs[0]
        # elif isinstance(op, RotaryEmbeddingOp):
        #     key = fused_op_extra_outputs[0][0]
        if isinstance(op, Bias) and op.return_bias:
            bias = fused_op_extra_outputs[0][0]
            get_bias = True

        x.requires_grad_(requires_grad=requires_grad)
        for idx, ys in zip(basic_op_idxs, fused_op_extra_outputs):
            for y in ys:
                if y is not None:
                    y.requires_grad_(requires_grad=requires_grad)
    
    # if profile_ao_ag:
    #     current_stream.synchronize()
    # if comm_op_fwd is not None:
    #     comm_op_fwd.sync()
    WAIT_EVENT.record(comm_op_fwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)

    assert get_bias == True
    return x, bias, comm_input


def run_partition_o_ar_backward(
    basic_op_ctxs,
    grad_out_1,
    grad_out_2,
    grad_bias_1,
    grad_bias_2,
    grad_params,
    backward_ops,
    comm_op_bwd,
    comm_overlap_window,
    comm_sm_configs,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_bwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_bwd is not None:
        # current_stream.synchronize()
        comm_op_bwd.event_record(current_stream)
        comm_op_bwd.event_wait()
        comm_op_bwd.fuser_forward(
            [None], grad_out_1,
            basic_op_extra_inputs=[], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size, 
                "backward": True
            }]
        )
        # comm_op_bwd.sync()
        WAIT_EVENT.record(comm_op_bwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)

    # if not profile:
    if comm_start == 0:
        comm_op_bwd.event_record(current_stream)

    # Apply backward ops
    dx = grad_out_2
    
    for fused_idx, (op, basic_op_idxs) in enumerate(backward_ops):

        # Stop if no more gradients are required
        if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
            dx = None
            break

        # Get extra input gradients based on operation type
        grad_extra_outputs = [()]
        if isinstance(op, Bias) and op.return_bias:
            grad_extra_outputs = [(grad_bias_2,)]

        if comm_start == fused_idx:
            # if not profile:
            comm_op_bwd.event_wait()
            # else:
            #     current_stream.synchronize()
            comm_op_bwd.fuser_forward(
                [None], grad_out_1,
                basic_op_extra_inputs=[], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size, 
                    "backward": True
                }]
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

        # if not profile:
        if fused_idx == comm_start - 1:
            comm_op_bwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_bwd.sync()
    
    # if comm_op_bwd is not None:
    #     comm_op_bwd.sync()
    WAIT_EVENT.record(comm_op_bwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)
    
    return grad_out_1, dx


def run_partition_o_ag_backward(
    basic_op_ctxs,
    grad_out_1,
    grad_bias_1,
    ag_k,
    ag_v,
    grad_params,
    backward_ops,
    comm_op_bwd,
    comm_overlap_window,
    comm_sm_configs,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_bwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_bwd is not None:
        # current_stream.synchronize()
        comm_op_bwd.fuser_forward(
            [None], ag_k,
            basic_op_extra_inputs=[(ag_v,)], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size, 
                "backward": True
            }]
        )
        # comm_op_bwd.sync()
        WAIT_EVENT.record(comm_op_bwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)

    # if not profile:
    # if comm_start == 0:
    #     comm_op_bwd.event_record(current_stream)

    # Apply backward ops
    dx = grad_out_1

    for fused_idx, (op, basic_op_idxs) in enumerate(backward_ops):

        # Stop if no more gradients are required
        if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
            dx = None
            break

        # Get extra input gradients based on operation type
        grad_extra_outputs = [()]
        if isinstance(op, Bias) and op.return_bias:
            grad_extra_outputs = [(grad_bias_1,)]

        if comm_start == fused_idx:
            # if not profile:
            # comm_op_bwd.event_wait()
            # else:
            #     current_stream.synchronize()
            comm_op_bwd.fuser_forward(
                [None], ag_k,
                basic_op_extra_inputs=[(ag_v,)], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size, 
                    "backward": True
                }]
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

        # if not profile:
        # if fused_idx == comm_start - 1:
        #     comm_op_bwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_bwd.sync()
    
    # if comm_op_bwd is not None:
    #     comm_op_bwd.sync()
    WAIT_EVENT.record(comm_op_bwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)
    
    return dx


def run_partition_a_ag_backward(
    basic_op_ctxs,
    grad_out_2,
    ag_k,
    ag_v,
    grad_params,
    backward_ops,
    comm_op_bwd,
    comm_overlap_window,
    comm_sm_configs,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_bwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_bwd is not None:
        # current_stream.synchronize()
        comm_op_bwd.fuser_forward(
            [None], ag_k,
            basic_op_extra_inputs=[(ag_v,)], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size, 
                "backward": True
            }]
        )
        # comm_op_bwd.sync()
        WAIT_EVENT.record(comm_op_bwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)

    # if not profile:
    # if comm_start == 0:
    #     comm_op_bwd.event_record(current_stream)

    # Apply backward ops
    dx = grad_out_2

    for fused_idx, (op, basic_op_idxs) in enumerate(backward_ops):

        # Stop if no more gradients are required
        if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
            dx = None
            break

        # Get extra input gradients based on operation type
        grad_extra_outputs = [()]

        if comm_start == fused_idx:
            # if not profile:
            # comm_op_bwd.event_wait()
            # else:
            #     current_stream.synchronize()
            comm_op_bwd.fuser_forward(
                [None], ag_k,
                basic_op_extra_inputs=[(ag_v,)], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size, 
                    "backward": True
                }]
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

        # if not profile:
        # if fused_idx == comm_start - 1:
        #     comm_op_bwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_bwd.sync()

        if isinstance(op, DotProductAttentionOp):
            grad_key, grad_value = fused_op_grad_extra_inputs[0]  # read from global variables
    
    # if comm_op_bwd is not None:
    #     comm_op_bwd.sync()
    WAIT_EVENT.record(comm_op_bwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)
    
    return dx


def run_partition_a_rs_backward(
    basic_op_ctxs,
    grad_out_1,
    grad_params,
    backward_ops,
    comm_op_bwd,
    comm_overlap_window,
    comm_sm_configs,
):
    current_stream = torch.cuda.current_stream()
    if comm_op_bwd:
        comm_start, comm_end = comm_overlap_window
        if comm_sm_configs:
            sm_num, block_size = comm_sm_configs
        else:
            sm_num, block_size = None, None
    else:
        comm_start, comm_end = -1, -1

    if comm_start == -1 and comm_op_bwd is not None:
        # current_stream.synchronize()
        grad_comm_key, grad_comm_value = comm_op_bwd.fuser_forward(
            [None], None, # read from global variables
            basic_op_extra_inputs=[(None,)], 
            basic_op_prev_ops=[None], 
            basic_op_next_ops=[None], 
            basic_op_kwargs=[{
                "sm_num": sm_num, 
                "block_size": block_size, 
                "backward": True
            }]
        )
        grad_comm_value = grad_comm_value[0][0]
        # comm_op_bwd.sync()
        WAIT_EVENT.record(comm_op_bwd.comm_stream)
        current_stream.wait_event(WAIT_EVENT)

    # if not profile:
    # if comm_start == 0:
    #     comm_op_bwd.event_record(current_stream)

    # Apply backward ops
    dx = grad_out_1

    for fused_idx, (op, basic_op_idxs) in enumerate(backward_ops):

        # Stop if no more gradients are required
        if all(not basic_op_ctxs[idx].requires_grad for idx in basic_op_idxs):
            dx = None
            break

        # Get extra input gradients based on operation type
        grad_extra_outputs = [()]

        if comm_start == fused_idx:
            # if not profile:
            # comm_op_bwd.event_wait()
            # else:
            #     current_stream.synchronize()
            grad_comm_key, grad_comm_value = comm_op_bwd.fuser_forward(
                [None], None, # read from global variables
                basic_op_extra_inputs=[(None,)], 
                basic_op_prev_ops=[None], 
                basic_op_next_ops=[None], 
                basic_op_kwargs=[{
                    "sm_num": sm_num, 
                    "block_size": block_size, 
                    "backward": True
                }]
            )
            grad_comm_value = grad_comm_value[0][0]

        # Backward op
        dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.fuser_backward(
            [basic_op_ctxs[idx] for idx in basic_op_idxs],
            dx,
            basic_op_grad_extra_outputs=grad_extra_outputs,
        )
        for idx, dparams in zip(basic_op_idxs, fused_op_grad_params):
            grad_params[idx] = dparams
            basic_op_ctxs[idx].saved_tensors = None

        # if not profile:
        # if fused_idx == comm_start - 1:
        #     comm_op_bwd.event_record(current_stream)

        # if comm_end == fused_idx:
        #     comm_op_bwd.sync()

        if isinstance(op, DotProductAttentionOp):
            grad_key, grad_value = fused_op_grad_extra_inputs[0]  # read from global variables
    
    # if comm_op_bwd is not None:
    #     comm_op_bwd.sync()
    WAIT_EVENT.record(comm_op_bwd.comm_stream)
    current_stream.wait_event(WAIT_EVENT)
    
    return dx, grad_comm_key, grad_comm_value


class _AttnOprojFuserAutogradFunction(torch.autograd.Function):
    """Autograd function for a pipeline of operations

    Autograd must be done at the pipeline level since we may apply
    different fusions in the forward and backward passes.

    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
        func_ctx: Optional[torch.autograd.function.FunctionCtx],
        query_1: torch.Tensor,
        query_2: torch.Tensor,
        comm_key: Optional[torch.Tensor],
        comm_value: Optional[torch.Tensor],
        comm_overlap_windows: Optional[List[Tuple[int, int]]],
        comm_sm_configs: Optional[List[Tuple[int, int]]],
        is_first_attn: bool,
        is_last_mlp: bool,
        profile_info: List[bool],
        comm_ops_fwd: Optional[List[FusibleOperation]],
        comm_ops_bwd: Optional[List[FusibleOperation]],
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
        basic_op_ctxs_1 = [OperationContext() for _ in range(len(basic_ops))]
        basic_op_ctxs_2 = [OperationContext() for _ in range(len(basic_ops))]
        
        profile_ao_ag, profile_ao_ar, profile_a_rs, profile_a_ag, profile_o_ag, profile_o_ar = profile_info

        if not profile_ao_ar:
            x, bias_1 = run_partition_ao_ag(
                basic_op_ctxs_1,
                query_1,
                comm_key,
                comm_value,
                forward_ops, 
                comm_ops_fwd[0], 
                comm_overlap_windows[0], 
                comm_sm_configs[0],
                is_grad_enabled,
                profile_ao_ag,
            )
            if profile_ao_ag:
                return x, bias_1
        
        if profile_ao_ar:
            comm_input = comm_key
        else:
            comm_input = x
        x, bias_2, comm_out = run_partition_ao_ar(
            basic_op_ctxs_2,
            query_2,
            comm_input,
            forward_ops, 
            comm_ops_fwd[1], 
            comm_overlap_windows[1], 
            comm_sm_configs[1],
            is_grad_enabled,
            profile_ao_ar,
        )
        if profile_ao_ar:
            return x, bias_2, comm_out
        
        out_1 = comm_out
        out_2 = x
        
        basic_op_ctxs = basic_op_ctxs_1 + basic_op_ctxs_2
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
            func_ctx.is_first_attn = is_first_attn
            func_ctx.profile_info = profile_info[2:]
            func_ctx.comm_overlap_windows = comm_overlap_windows[2:]
            func_ctx.comm_sm_configs = comm_sm_configs[2:]
            func_ctx.comm_ops_bwd = comm_ops_bwd
            func_ctx.backward_ops = backward_ops
            func_ctx.basic_ops = basic_ops
            func_ctx.basic_op_ctxs = basic_op_ctxs
            func_ctx.basic_op_num_params = [sum(1 for _ in op.parameters()) for op in basic_ops]
        
        if is_last_mlp:
            comm_ops_fwd[1].fuser_forward(
                [None], None,
                basic_op_extra_inputs=[], basic_op_prev_ops=[None], basic_op_next_ops=[None], 
                basic_op_kwargs=[{"sm_num": 30, "block_size": 1024}]
            )
            comm_ops_fwd[1].sync()

        if profile_a_rs and profile_a_ag and profile_o_ag and profile_o_ar:
            return out_1, out_2, bias_1, bias_2, func_ctx
        else:
            return out_1, out_2, bias_1, bias_2

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx: Any,
        grad_out_1: torch.Tensor,
        grad_out_2: torch.Tensor,
        grad_bias_1: torch.Tensor,
        grad_bias_2: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Backward pass"""

        # Operations and autograd state
        is_first_attn = func_ctx.is_first_attn
        profile_info = func_ctx.profile_info
        comm_ops_bwd = func_ctx.comm_ops_bwd
        comm_overlap_windows = func_ctx.comm_overlap_windows
        comm_sm_configs = func_ctx.comm_sm_configs
        backward_ops = func_ctx.backward_ops
        basic_ops = func_ctx.basic_ops
        basic_op_ctxs = func_ctx.basic_op_ctxs

        # Unflatten list of saved tensors
        for i, ctx in enumerate(basic_op_ctxs):
            if ctx._saved_tensors_range is not None:
                ctx.saved_tensors = func_ctx.saved_tensors[slice(*ctx._saved_tensors_range)]
        basic_op_ctxs_1 = basic_op_ctxs[:len(basic_ops)]
        basic_op_ctxs_2 = basic_op_ctxs[len(basic_ops):]

        grad_params_1 = [None for _ in range(len(basic_ops))]
        grad_params_2 = [None for _ in range(len(basic_ops))]
    
        profile_a_rs, profile_a_ag, profile_o_ag, profile_o_ar = profile_info

        if not profile_a_rs and not profile_a_ag and not profile_o_ag:
            grad_out_1, grad_out_2 = run_partition_o_ar_backward(
                basic_op_ctxs_2,
                grad_out_1,
                grad_out_2,
                grad_bias_1,
                grad_bias_2,
                grad_params_2,
                backward_ops[:2], # Bias, BasicLinear, DotProductAttention
                comm_ops_bwd[-1], # RS, AG, AG, AR
                comm_overlap_windows[-1],
                comm_sm_configs[-1],
            )
            if profile_o_ar:
                return grad_out_1, grad_out_2
        
        if not profile_a_rs and not profile_a_ag:
            attn_ctx = basic_op_ctxs[0]
            k_pre, v_pre = attn_ctx.saved_tensors[1], attn_ctx.saved_tensors[2]
            grad_out_1 = run_partition_o_ag_backward(
                basic_op_ctxs_1,
                grad_out_1,
                grad_bias_1,
                k_pre,
                v_pre,
                grad_params_1,
                backward_ops[:2], # Bias, BasicLinear, DotProductAttention
                comm_ops_bwd[-2], # RS, AG, AG, AR
                comm_overlap_windows[-2],
                comm_sm_configs[-2],
            )
            if profile_o_ag:
                return grad_out_1
        
        if not profile_a_rs:
            attn_ctx = basic_op_ctxs[0]
            k_pre, v_pre = attn_ctx.saved_tensors[1], attn_ctx.saved_tensors[2]
            grad_query_2 = run_partition_a_ag_backward(
                basic_op_ctxs_2,
                grad_out_2,
                k_pre,
                v_pre,
                grad_params_2,
                [backward_ops[2]], # Bias, BasicLinear, DotProductAttention
                comm_ops_bwd[1], # RS, AG, AG, AR
                comm_overlap_windows[1],
                comm_sm_configs[1],
            )
            if profile_a_ag:
                return grad_query_2
        
        grad_query_1, grad_comm_key, grad_comm_value = run_partition_a_rs_backward(
            basic_op_ctxs_1,
            grad_out_1,
            grad_params_1,
            [backward_ops[2]], # Bias, BasicLinear, DotProductAttention
            comm_ops_bwd[0], # RS, AG, AG, AR
            comm_overlap_windows[0],
            comm_sm_configs[0],
        )
        if profile_a_rs:
            return grad_query_1, grad_comm_key, grad_comm_value

        # Flatten list of parameter gradients
        grad_params_flat = []
        for idx, (dparams_1, dparams_2) in enumerate(zip(grad_params_1, grad_params_2)):
            num_params = func_ctx.basic_op_num_params[idx]
            dparams_1 = [None] * num_params if dparams_1 is None else list(dparams_1)
            dparams_2 = [None] * num_params if dparams_2 is None else list(dparams_2)

            if len(dparams_1) != num_params or len(dparams_2) != num_params:
                if not (profile_a_rs or profile_a_ag or profile_o_ag or profile_o_ar):
                    raise RuntimeError(
                        f"Expected op {idx} to generate {num_params} param grads, "
                        f"but got {len(dparams_1)} and {len(dparams_2)}"
                    )
                dparams_1 = dparams_2 = [None] * num_params

            dparams = []
            # Sum the gradients of the two nanobatches
            for dparam_1, dparam_2 in zip(dparams_1, dparams_2):
                dparam = None if dparam_1 is None and dparam_2 is None else \
                        dparam_2 if dparam_1 is None else \
                        dparam_1 if dparam_2 is None else \
                        dparam_1.add_(dparam_2)
                dparams.append(dparam)
            grad_params_flat.extend(dparams)

        return (
            grad_query_1,
            grad_query_2,
            grad_comm_key,
            grad_comm_value,
            None,  # comm_overlap_windows
            None,  # comm_sm_configs
            None,  # is_first_attn
            None,  # is_last_mlp
            None,  # profile_info
            None,  # comm_ops_fwd
            None,  # comm_ops_bwd
            None,  # forward_ops
            None,  # backward_ops
            None,  # basic_ops
            None,  # is_grad_enabled
            None,  # num_params
            *grad_params_flat,
        )


class AttnOprojPartitionFuser:
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
        comm_ops_fwd: Optional[List[FusibleOperation]] = None,
        comm_ops_bwd: Optional[List[FusibleOperation]] = None,
        fuse_ops: bool = True,   
        is_first_attn: bool = False,
        is_last_mlp: bool = False,
        profile_ao_ag: bool = False,
        profile_ao_ar: bool = False,
        profile_a_rs: bool = False,
        profile_a_ag: bool = False,
        profile_o_ag: bool = False,
        profile_o_ar: bool = False,
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

        self._comm_ops_fwd = comm_ops_fwd
        self._comm_ops_bwd = comm_ops_bwd if comm_ops_bwd is not None else comm_ops_fwd

        # Fuse ops if needed
        if fuse_ops:
            self.fuse_ops()
        
        self._is_first_attn = is_first_attn
        self._is_last_mlp = is_last_mlp
        self._profile_info = [
            profile_ao_ag, profile_ao_ar, profile_a_rs, profile_a_ag, profile_o_ag, profile_o_ar
        ]

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
        query_1: torch.Tensor,
        query_2: torch.Tensor,
        comm_key: Optional[torch.Tensor] = None,
        comm_value: Optional[torch.Tensor] = None,
        comm_overlap_window_ao_ag: Optional[Tuple[int, int]] = None,
        comm_sm_configs_ao_ag: Optional[Tuple[int, int]] = None,
        comm_overlap_window_ao_ar: Optional[Tuple[int, int]] = None,
        comm_sm_configs_ao_ar: Optional[Tuple[int, int]] = None,
        comm_overlap_window_a_rs: Optional[Tuple[int, int]] = None,
        comm_sm_configs_a_rs: Optional[Tuple[int, int]] = None,
        comm_overlap_window_a_ag: Optional[Tuple[int, int]] = None,
        comm_sm_configs_a_ag: Optional[Tuple[int, int]] = None,
        comm_overlap_window_o_ag: Optional[Tuple[int, int]] = None,
        comm_sm_configs_o_ag: Optional[Tuple[int, int]] = None,
        comm_overlap_window_o_ar: Optional[Tuple[int, int]] = None,
        comm_sm_configs_o_ar: Optional[Tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, ...]: # hidden_states, bias, residual

        # Initialization before forward pass
        for op in self._basic_ops:
            op.pre_forward()

        # Flatten list of parameters
        params = [param for op in self._basic_ops for param in op.parameters()]
        comm_overlap_windows = [
            comm_overlap_window_ao_ag, 
            comm_overlap_window_ao_ar, 
            comm_overlap_window_a_rs, 
            comm_overlap_window_a_ag, 
            comm_overlap_window_o_ag, 
            comm_overlap_window_o_ar,
        ]
        comm_sm_configs = [
            comm_sm_configs_ao_ag, 
            comm_sm_configs_ao_ar, 
            comm_sm_configs_a_rs, 
            comm_sm_configs_a_ag, 
            comm_sm_configs_o_ag, 
            comm_sm_configs_o_ar,
        ]

        # Fuser forward pass
        is_grad_enabled = torch.is_grad_enabled()
        if is_grad_enabled:
            forward_func = _AttnOprojFuserAutogradFunction.apply
            args = []
        else:
            forward_func = _AttnOprojFuserAutogradFunction.forward
            args = [None]
        args += (
            query_1,
            query_2,
            comm_key,
            comm_value,
            comm_overlap_windows,
            comm_sm_configs,
            self._is_first_attn,
            self._is_last_mlp,
            self._profile_info,
            self._comm_ops_fwd,
            self._comm_ops_bwd,
            self._forward_ops,
            self._backward_ops,
            self._basic_ops,
            is_grad_enabled,
            len(params),
            *params,
        )
        return forward_func(*args)
