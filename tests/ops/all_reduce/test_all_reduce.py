#!/usr/bin/env python3
"""
Test script for the attention fuser with the required operations:
- BDA (BiasDropoutAddOp)
- LayerNorm
- Linear_qkv (Linear transformation for queries, keys, values)
- post_process_qkv (QKVPostProcessOp)
- rotary embedding (RotaryEmbeddingOp)
- linear_proj (Linear projection for output)
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sys
import os
import pytest
from typing import Optional, Tuple
import argparse
import random
import time

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import required operations
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.qkv_postprocess_op import create_qkv_postprocess_op
from kareus.megatron.core.extensions.rotary_embedding_op import create_rotary_embedding_op
from kareus.transformer_engine.pytorch.attention.dot_product_attention import DotProductAttentionOp
from kareus.megatron.core.extensions.te_linear import TEFusibleColumnParallelLinear, TEFusibleRowParallelLinear, TEFusibleLinear
from kareus.megatron.core.extensions.te_attention import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
# Import attention fuser
from kareus.megatron.core.extensions.attention_fuser import AttentionFuser

# Import configuration
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from cfuser.core.utils import nvtx_range

from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_distributed(tensor_parallel_size: int = 1, backend: str = 'nccl'):
    """Initialize distributed processing for tensor parallelism.
    
    Parameters
    ----------
    tensor_parallel_size : int
        Size of tensor parallel group
    backend : str
        Distributed backend to use ('nccl', 'gloo', etc.)
        
    Returns
    -------
    torch.distributed.ProcessGroup or None
        Tensor parallel process group, or None if single process
    """
    if tensor_parallel_size <= 1:
        print("Single process mode - no distributed initialization needed")
        return None
        
    # Initialize the process group if not already initialized
    if not dist.is_initialized():
        # For testing, we'll use a single machine setup
        # In real scenarios, you'd have proper rank/world_size from environment
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', tensor_parallel_size))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set the device before initializing distributed
        torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        print(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Create tensor parallel group
    if tensor_parallel_size > 1:
        # Create process groups for tensor parallelism
        ranks = list(range(tensor_parallel_size))
        tp_group = dist.new_group(ranks)
        print(f"Created tensor parallel group with ranks: {ranks}")
        return tp_group
    
    return None


class AllReduceTest:
    """Test suite for attention fuser operations."""

    def __init__(self, device='cuda', tensor_parallel_size: int = 1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.tensor_parallel_size = tensor_parallel_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(tensor_parallel_size)
        
        # Test configuration
        self.batch_size = 4
        self.seq_length = 4096
        self.hidden_size = 2048
        self.num_attention_heads = 32
        self.num_query_groups = 8  # For grouped query attention
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Create transformer config
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_query_groups=self.num_query_groups,
            layernorm_epsilon=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            qk_layernorm=False,
            apply_query_key_layer_scaling=False,
            rotary_interleaved=False,
            flash_decode=False,
            apply_rope_fusion=True,
            params_dtype=self.dtype,
        )

    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        allreduce_inputs = torch.randn(
            self.seq_length, self.batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        return allreduce_inputs

    def create_operations(self, rank: int, world_size: int, allreduce_inputs: torch.Tensor):
        """Create all the required operations for the attention fuser."""
        allreduce_msccl_cached = AllReduce(
            process_group=self.tp_group,
            async_op=True,
            backend="msccl",
            rank=rank,
            world_size=world_size,
        )
        allreduce_msccl = AllReduce(
            process_group=self.tp_group,
            async_op=True,
            backend="msccl",
            rank=rank,
            world_size=world_size,
            use_persistent_output=True,
            input_buffer=allreduce_inputs,
            tensor_size=[self.seq_length, self.batch_size, self.hidden_size],
            device=self.device,
            dtype=self.dtype,
        )
        allreduce_nccl = AllReduce(
            process_group=self.tp_group,
            async_op=True,
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        return allreduce_msccl_cached, allreduce_msccl, allreduce_nccl

    def test_individual_operations(self, allreduce_comm_op, allreduce_inputs, sm_num=None, block_size=None):
        """Test each operation individually to ensure they work correctly."""
        allreduce_output = allreduce_comm_op(allreduce_inputs, sm_num=sm_num, block_size=block_size)
        allreduce_comm_op.sync()
        return True

    def run_all_tests(self, rank: int, world_size: int):
        allreduce_inputs = self.create_test_tensors()
        allreduce_msccl_cached, allreduce_msccl, allreduce_nccl = self.create_operations(rank, world_size, allreduce_inputs)

        # for sm_num in range(1, 21):
        #     block_size = 1024
        sm_num = 8
        block_size = 1024
        with nvtx_range(f"nccl - {sm_num} - {block_size}"):
            for i in range(5):
                self.test_individual_operations(allreduce_nccl, allreduce_inputs, sm_num=sm_num, block_size=block_size)

            # start_time = time.perf_counter()
            # for i in range(10):
            #     self.test_individual_operations(allreduce_nccl, allreduce_inputs)
            # end_time = time.perf_counter()
            # print(f"Time taken for nccl with sm_num={sm_num} and block_size={block_size}: {(end_time - start_time) / 10} seconds")

        for sm_num in range(1, 21):
            block_size = 1024
            with nvtx_range(f"msccl cached - {sm_num}"):
                for i in range(5):
                    self.test_individual_operations(allreduce_msccl_cached, allreduce_inputs, sm_num=sm_num, block_size=block_size)
            
            with nvtx_range(f"msccl - {sm_num}"):
                for i in range(5):
                    self.test_individual_operations(allreduce_msccl, allreduce_inputs, sm_num=sm_num, block_size=block_size)
            
            # start_time = time.perf_counter()
            # for i in range(10):
            #     self.test_individual_operations(allreduce_msccl, allreduce_inputs, sm_num=sm_num, block_size=block_size)
            # end_time = time.perf_counter()
            # print(f"Time taken for msccl with sm_num={sm_num} and block_size={block_size}: {(end_time - start_time) / 10} seconds")
        

        # self.test_individual_operations(allreduce_msccl, allreduce_inputs, sm_num=4, block_size=1024)
        # self.test_individual_operations(allreduce_nccl, allreduce_inputs, sm_num=4, block_size=1024)
        return True


def run_process(rank, world_size, args, master_port):
    """Run the attention fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = AllReduceTest(
        device=args.device, 
        tensor_parallel_size=args.tensor_parallel_size
    )
    success = test_runner.run_all_tests(rank, world_size)

    # Clean up distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


def main():
    """Main function to run the attention fuser tests."""
    parser = argparse.ArgumentParser(description='Attention Fuser Test with Distributed Support')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run tests on')
    parser.add_argument('--tensor-parallel-size', type=int, default=2, 
                        help='Tensor parallel size for distributed testing')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Distributed backend (nccl, gloo)')
    
    args = parser.parse_args()
    
    print(f"Running tests with:")
    print(f"  Device: {args.device}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"  Backend: {args.backend}")

    from torch.multiprocessing import spawn
    spawn(
        run_process,
        args=(
            args.tensor_parallel_size,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=args.tensor_parallel_size,
        join=True,
    )


if __name__ == "__main__":
    exit(main())
