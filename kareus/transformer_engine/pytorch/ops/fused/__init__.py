"""
Modified from TransformerEngine (transformer_engine/pytorch/ops/fused/__init__.py).
Changes:
- Only exports ``fuse_forward_linear_bias_activation``; the original
  also exports ``fuse_forward_linear_bias_add``,
  ``fuse_backward_linear_add``, and other fused-op constructors that are
  not needed by the partition scheduler.
"""

from .forward_linear_bias_activation import fuse_forward_linear_bias_activation