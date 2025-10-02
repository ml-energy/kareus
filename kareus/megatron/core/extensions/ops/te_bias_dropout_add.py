# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Transformer Engine wrapper for BiasDropoutAddOp with Megatron interface compatibility."""

from typing import Callable, Optional, Tuple
import torch

from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp


def _bias_dropout_add_te_training(x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float) -> torch.Tensor:
    """TE BiasDropoutAddOp wrapper for training mode."""
    x, bias = x_with_bias  # unpack
    
    # Create the operation instance for training
    op_module = BiasDropoutAddOp(dropout_prob=prob, training=True)
    
    # Call the operation
    return op_module(x, bias, residual)


def _bias_dropout_add_te_inference(x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float) -> torch.Tensor:
    """TE BiasDropoutAddOp wrapper for inference mode."""
    x, bias = x_with_bias  # unpack
    
    # Create the operation instance for inference
    op_module = BiasDropoutAddOp(dropout_prob=prob, training=False)
    
    # Call the operation
    return op_module(x, bias, residual)


def te_fusible_get_bias_dropout_add(training: bool, fused: bool) -> Callable:
    """Get bias dropout add function with the same interface as Megatron.
    
    Args:
        training (bool): Whether the model is in training mode
        fused (bool): Whether to use fused implementation (must be True)
        
    Returns:
        Callable: Function with signature (x_with_bias, residual, prob) -> output
        
    Raises:
        AssertionError: If fused is not True
    """
    assert fused, "BiasDropoutAddOp only supports fused=True"
    
    if training:
        return _bias_dropout_add_te_training
    else:
        return _bias_dropout_add_te_inference