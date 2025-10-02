# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations.

This operation-based API is experimental and subject to change.

"""

from kareus.transformer_engine.pytorch.ops.basic import *
from kareus.transformer_engine.pytorch.ops.linear import Linear

# Global variables for storing gathered K and V in context parallelism
# These are used to pass data in backward passes
K_AG = None
V_AG = None
