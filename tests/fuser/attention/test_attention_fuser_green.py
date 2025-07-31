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

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.te_attention import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
from kareus.megatron.core.extensions.attention_fuser_green import AttentionFuser
from kareus.megatron.core.extensions.qkv_postprocess_op import QKVPostProcessOp
from kareus.megatron.core.extensions.rotary_embedding_op import RotaryEmbeddingOp

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from cfuser.core.utils import nvtx_range
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_distributed(rank, world_size, backend: str = 'nccl'):
    if world_size <= 1:
        return None
        
    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        print(f"Initialized distributed: rank={rank}, world_size={world_size}")
    
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks)
    print(f"Created tensor parallel group with ranks: {ranks}")
    return tp_group


class AttentionFuserTest:
    """Test suite for attention fuser operations."""

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size
        
        # Set the current CUDA device for this rank
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        
        # Initialize distributed processing
        self.tp_group = init_distributed(rank, world_size)
        
        # Test configuration
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = 2048
        self.num_attention_heads = 32
        self.num_query_groups = 32  # For grouped query attention
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.ffn_hidden_size = 4 * self.hidden_size
        
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
            tensor_model_parallel_size=world_size,
        )

    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        nano_batch_size = self.batch_size // 2
        hidden_states = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        bias = torch.randn(
            self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        residual = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        seq = (
            torch.arange(self.seq_length, device=self.device, dtype=torch.float32)
            + 0
        )
        rotary_base = 10000
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.outer(seq, inv_freq)
        rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        
        attention_mask = None

        allreduce_inputs = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        return hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs

    def create_operations(self):
        """Create all the required operations for the attention fuser."""
        
        # 1. BDA Operation (Bias Dropout Add)
        bda_op = BiasDropoutAddOp(
            dropout_prob=self.config.hidden_dropout,
            training=True
        )
        
        # 2. LayerNorm Operation
        layernorm_op = LayerNorm(
            normalized_shape=self.hidden_size,
            eps=self.config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype
        )
        
        # 3. Linear QKV Operation (transforms input to queries, keys, values)
        qkv_hidden_size = (
            self.num_attention_heads * self.head_dim +  # Query heads
            self.num_query_groups * self.head_dim +     # Key heads
            self.num_query_groups * self.head_dim       # Value heads
        )
        qkv_hidden_size = qkv_hidden_size // self.tensor_parallel_size
        linear_qkv_op = Linear(
            in_features=self.hidden_size,
            out_features=qkv_hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=False,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        
        # 4. QKV Post-process Operation
        num_query_groups_per_partition = self.num_query_groups // self.tensor_parallel_size
        num_attention_heads_per_partition = self.num_attention_heads // self.tensor_parallel_size
        qkv_postprocess_op = QKVPostProcessOp(
            num_query_groups_per_partition=num_query_groups_per_partition,
            num_attention_heads_per_partition=num_attention_heads_per_partition,
            hidden_size_per_attention_head=self.head_dim,
            q_layernorm=None,
            k_layernorm=None,
            run_tests_fn=None,
            test_mode=False,
        ) 
        
        # 5. Rotary Embedding Operation
        rotary_embedding_op = RotaryEmbeddingOp(
            config=self.config,
        ) 
        
        # 6. Dot Product Attention Operation
        attention_op = TEFusibleDotProductAttention(
            config=self.config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        
        # 7. Linear Projection Operation
        hidden_size_in = self.hidden_size // self.tensor_parallel_size
        linear_proj_op = Linear(
            in_features=hidden_size_in,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=True,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        
        # 8. AllReduce Communication Operation
        if self.tensor_parallel_size > 1:
            allreduce_comm_op = AllReduce(
                process_group=self.tp_group,
                async_op=True,  # Use async mode as set in the modified AllReduce
                backend="nccl",
                # rank=self.rank,
                # world_size=self.world_size,
            )
        
            return [
                bda_op,
                layernorm_op,
                linear_qkv_op,
                qkv_postprocess_op,
                rotary_embedding_op,
                attention_op,
                linear_proj_op,
                allreduce_comm_op
            ]

        else:
            # For single process mode, create a dummy AllReduce that does nothing
            # This allows us to test the fuser logic without actual communication
            print("Warning: Single process mode - AllReduce will be a no-op")
            
            class DummyAllReduce:
                def set_stream(self, stream): pass
                def fuser_forward(self, *args, **kwargs): pass
                def event_wait(self): pass
                def event_record(self, stream): pass
                def sync(self): pass
            
            dummy_allreduce = DummyAllReduce()
            
            return [
                bda_op,
                layernorm_op,
                linear_qkv_op,
                qkv_postprocess_op,
                rotary_embedding_op,
                attention_op,
                linear_proj_op,
                dummy_allreduce
            ]

    def test_attention_fuser(self, rank: int, world_size: int):
        test_tensors = self.create_test_tensors()
        operations = self.create_operations()

        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = test_tensors
        comp_ops = operations[:7]
        allreduce_comm_op = operations[7]

        # Create green context streams for compute and communication
        device = self.device
        
        # Check GPU capabilities first
        gpu_props = torch.cuda.get_device_properties(device)
        total_sms = gpu_props.multi_processor_count
        compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
        
        print(f"GPU: {gpu_props.name}")
        print(f"Total SMs: {total_sms}")
        print(f"Compute capability: {compute_capability}")
        
        # Check if green contexts are supported (requires compute capability 9.0+)
        if gpu_props.major < 9:
            print(f"Warning: Green contexts require compute capability 9.0+, but got {compute_capability}")
            print("Falling back to regular streams...")
            # Use regular CUDA streams as fallback
            compute_stream = torch.cuda.Stream()
            communication_stream = torch.cuda.Stream()
            resources = None
        else:
            # Use a more conservative SM allocation
            compute_sms = max(8, total_sms // 2)  # At least 8 SMs for compute
            comm_sms = max(4, total_sms // 4)     # At least 4 SMs for communication
            
            # Ensure we don't exceed total SMs
            if compute_sms + comm_sms > total_sms:
                compute_sms = total_sms - comm_sms
                
            sm_counts = [compute_sms, comm_sms]
            print(f"Requesting SM allocation: compute={compute_sms}, communication={comm_sms}")
            
            try:
                streams, resources = split_device_green_ctx_by_sm_count(device, sm_counts)
                compute_stream = streams[0]      # Stream for compute operations
                communication_stream = streams[1] # Stream for communication operations
            except RuntimeError as e:
                print(f"Green context allocation failed: {e}")
                print("Falling back to regular streams...")
                # Fallback to regular streams
                compute_stream = torch.cuda.Stream()
                communication_stream = torch.cuda.Stream()
                resources = None
        
        if resources is not None:
            print(f"Created green context streams:")
            print(f"  Compute stream SMs: {resources[0].sm.smCount}")
            print(f"  Communication stream SMs: {resources[1].sm.smCount}")
            print(f"  Remaining SMs: {resources[2].sm.smCount}")
        else:
            print(f"Using regular CUDA streams (no SM partitioning)")

        attention_fuser = AttentionFuser(
            ops=comp_ops,
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=False,
            compute_stream=compute_stream,
            communication_stream=communication_stream
        )

        overlap_window = (-1, -1)
        sm_configs = (4, 1024)

        import signal
        import time
        
        def timeout_handler(signum, frame):
            print(f"TIMEOUT: Process {rank} timed out - possible deadlock")
            print("Stack trace at timeout:")
            import traceback
            traceback.print_stack(frame)
            raise TimeoutError("Operation timed out after 30 seconds")
        
        # Set up timeout for green context operations
        signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            for i in range(5):
                try:
                    signal.alarm(30)  # 30 second timeout
                    print(f"Process {rank}: Running iteration {i+1}/5...")
                    
                    # Add debug prints to identify hang location
                    print(f"Process {rank}: About to call attention_fuser...")
                    
                    # Ensure proper synchronization before each iteration
                    if resources is not None:  # Using green contexts
                        print(f"Process {rank}: Synchronizing green contexts...")
                        torch.cuda.synchronize()
                        # dist.barrier()
                    else:
                        print(f"Process {rank}: Synchronizing regular streams...")
                        compute_stream.synchronize()
                        communication_stream.synchronize()
                    
                    print(f"Process {rank}: Calling attention_fuser forward...")
                    
                    with torch.cuda.stream(compute_stream):
                        output, output_bias, output_residual, allreduce_output = attention_fuser(
                            hidden_states=hidden_states,
                            bias=bias,
                            residual=residual,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_mask=attention_mask,
                            allreduce_input=allreduce_inputs,
                            allreduce_overlap_window=overlap_window,
                            allreduce_sm_configs=sm_configs,
                        )
                    
                    print(f"Process {rank}: Attention fuser completed, synchronizing...")
                    
                    # Ensure all operations complete before next iteration
                    if resources is not None:  # Using green contexts
                        torch.cuda.synchronize()
                        # dist.barrier()
                    else:
                        compute_stream.synchronize()
                        communication_stream.synchronize()
                        
                    print(f"Process {rank}: Iteration {i+1} completed successfully")
                    signal.alarm(0)  # Cancel timeout
                    
                    # Add a small delay to prevent resource contention
                    time.sleep(0.1)
                    
                except TimeoutError as e:
                    print(f"Process {rank}: Iteration {i+1} timed out: {e}")
                    signal.alarm(0)
                    return False
                except Exception as e:
                    print(f"Process {rank}: Iteration {i+1} failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                    signal.alarm(0)
                    return False
                    
        finally:
            # Cleanup resources
            signal.alarm(0)
            try:
                if resources is not None:
                    print(f"Process {rank}: Cleaning up green context resources...")
                    # Clean up green context resources if needed
                    del resources
                    torch.cuda.synchronize()
                else:
                    print(f"Process {rank}: Cleaning up regular streams...")
                    compute_stream.synchronize()
                    communication_stream.synchronize()
            except Exception as e:
                print(f"Process {rank}: Error during cleanup: {e}")
        
        return True

    def run_all_tests(self, rank: int, world_size: int):
        self.test_attention_fuser(rank, world_size)
        return True


def run_process(rank, world_size, args, master_port):
    """Run the attention fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = AttentionFuserTest(args, rank, world_size)
    success = test_runner.run_all_tests(rank, world_size)

    # Clean up distributed if initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed process group destroyed")


def main():
    """Main function to run the attention fuser tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    parser.add_argument("--single_process", action="store_true", help="Run in single process mode for profiling")
    parser.add_argument("--use_torchrun", action="store_true", help="Use torchrun instead of multiprocessing (better for profiling)", default=False)
    args = parser.parse_args()

    print("Running overlap test for attention fuser")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Frequency: {args.frequency}")

    # if args.use_torchrun:
    #     # When using torchrun, get rank and world_size from environment
    #     import os
    #     rank = int(os.environ.get("RANK", 0))
    #     world_size = int(os.environ.get("WORLD_SIZE", 1))
    #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
    #     print(f"Using torchrun: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
    #     master_port = random.randint(8000, 65535)
    #     # Set master port if not set
    #     if "MASTER_PORT" not in os.environ:
    #         os.environ["MASTER_PORT"] = f"{master_port}"
    #     if "MASTER_ADDR" not in os.environ:
    #         os.environ["MASTER_ADDR"] = "localhost"
            
    #     run_process(rank, world_size, args, master_port)
        
    # elif args.single_process:
    #     print("Running in single process mode (for profiling)")
    #     # Set environment variables for single process
    #     os.environ["RANK"] = "0"
    #     os.environ["WORLD_SIZE"] = "1"
    #     os.environ["LOCAL_RANK"] = "0"
    #     os.environ["MASTER_ADDR"] = "localhost"
    #     os.environ["MASTER_PORT"] = "12345"
    #     # Run single process without multiprocessing
    #     run_process(0, 1, args, 12345)  # rank=0, world_size=1
    # else:
    from torch.multiprocessing import spawn
    spawn(
        run_process,
        args=(
            args.world_size,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    exit(main())
