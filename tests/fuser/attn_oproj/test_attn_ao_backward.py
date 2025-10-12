import os
import torch
import torch.distributed as dist
import time
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from megatron.core.transformer.transformer_config import TransformerConfig
from common_config import FuserTestConfig
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from kareus.megatron.core.extensions.ops import QKVPostProcessOp
from kareus.megatron.core.extensions.ops import RotaryEmbeddingOp
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.transformer_engine.pytorch.ops.basic.all_gather_kv import AllGatherKV, K_TO_SAVE, V_TO_SAVE, K_AG, V_AG
from kareus.transformer_engine.pytorch.ops.basic.reduce_scatter_kv import ReduceScatterKV, K_RS, V_RS
from kareus.megatron.core.extensions.ops import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
from kareus.megatron.core.extensions.attn_oproj_fuser import AttnOprojPartitionFuser as PartitionFuser
from megatron.core.transformer.enums import AttnMaskType
from zeus.monitor import ZeusMonitor
from kareus.utils.debug import nvtx_range
import pynvml
import multiprocessing


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
    group = dist.new_group(ranks)
    print(f"Created context parallel group with ranks: {ranks}")
    return group

class AttentionFuserTest:
    """Test suite for attention fuser operations."""

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.context_parallel_size = args.context_parallel_size
        self.tensor_parallel_size = args.tensor_parallel_size
        self.model_name = args.model_name
        
        # Initialize distributed processing
        self.cp_group = init_distributed(rank, world_size)
        self.tp_group = self.cp_group
        
        # Test configuration
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        self.head_dim = FuserTestConfig.HEAD_DIM
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE
        
        # Create transformer config
        self.config = FuserTestConfig.create_attention_config()
        if rank == 0:
            print(f"self.config: {self.config}")

        self.frequency = args.frequency
        self.repeat_num = 1

        self.overlap_window_forward = (-1, -1)
        self.sm_configs_forward = (20, 1024)
    
    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        nano_batch_size = self.batch_size // 2
        local_seq_length = self.seq_length // self.context_parallel_size
        local_num_attention_heads = self.num_attention_heads // self.tensor_parallel_size
        local_query_groups = self.num_query_groups // self.tensor_parallel_size

        query_1 = torch.randn(
            local_seq_length, nano_batch_size, local_num_attention_heads, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        query_2 = torch.randn(
            local_seq_length, nano_batch_size, local_num_attention_heads, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        key_to_save = torch.randn(
            local_seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        value_to_save = torch.randn(
            local_seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        global K_TO_SAVE, V_TO_SAVE
        K_TO_SAVE[0] = key_to_save
        V_TO_SAVE[0] = value_to_save

        k_ag = torch.randn(
            self.seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        v_ag = torch.randn(
            self.seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        global K_AG, V_AG
        K_AG[0] = k_ag
        V_AG[0] = v_ag

        allgather_key = torch.randn(
            local_seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        allgather_value = torch.randn(
            local_seq_length, nano_batch_size, local_query_groups, self.head_dim,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        allreduce_inputs = torch.randn(
            local_seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )
        
        return query_1, query_2, allgather_key, allgather_value, allreduce_inputs
    
    def create_gradient_tensors(self):
        """Create gradient tensors for backward pass testing."""
        nano_batch_size = self.batch_size // 2
        local_seq_length = self.seq_length // self.context_parallel_size
        local_attention_heads = self.num_attention_heads // self.tensor_parallel_size
        local_query_groups = self.num_query_groups // self.tensor_parallel_size
        
        output_grad_1 = torch.randn(
            local_seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )

        output_grad_2 = torch.randn(
            local_seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )
        
        bias_grad_1 = None
        bias_grad_2 = None
        
        return output_grad_1, output_grad_2, bias_grad_1, bias_grad_2
    
    def create_operations(self, allreduce_inputs):
        """Create all the required operations for the attention fuser."""     
        # 6. Dot Product Attention Operation
        attention_op = TEFusibleDotProductAttention(
            config=self.config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            cp_comm_type="all_gather",
        )
        attention_op.set_context_parallel_group(
            cp_group=self.cp_group,
            cp_global_ranks=list(range(self.world_size)),
            cp_stream=torch.cuda.Stream(),
        )

        # 7. Linear Projection Operation
        hidden_size_in = (self.head_dim * self.num_attention_heads) // self.tensor_parallel_size
        linear_proj_op = Linear(
            in_features=hidden_size_in,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=False,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )
        
        # 8. AllReduce Communication Operation
        nano_batch_size = self.batch_size // 2
        local_seq_length = self.seq_length // self.context_parallel_size
        local_query_groups = self.num_query_groups // self.tensor_parallel_size
        new_size = [self.seq_length, nano_batch_size, local_query_groups, self.head_dim]

        allgather_comm_op_1 = AllGatherKV(
            process_group=self.cp_group,
            async_op=True,
            backend="msccl",
            rank=self.rank,
            world_size=self.world_size,
            tensor_size=new_size,
            device=self.device,
            dtype=self.dtype,
            batch_idx=0,
        )

        allreduce_comm_op_1 = AllReduce(
            process_group=self.tp_group,
            async_op=True,
            backend="msccl",
            rank=self.rank,
            world_size=self.world_size,
            use_persistent_output=True,
            input_buffer=allreduce_inputs,
            tensor_size=[local_seq_length, nano_batch_size, self.hidden_size],
            device=self.device,
            dtype=self.dtype,
        )
        
        reducescatter_comm_op = ReduceScatterKV(
            process_group=self.cp_group,
            async_op=True,
            backend="msccl",
            rank=self.rank,
            world_size=self.world_size,
            tensor_size=new_size,
            device=self.device,
            dtype=self.dtype,
        )

        allgather_comm_op_2 = allgather_comm_op_1
        allgather_comm_op_3 = allgather_comm_op_1

        allreduce_comm_op_2 = allreduce_comm_op_1

        comm_ops = [
            allgather_comm_op_1,
            allreduce_comm_op_1,
            reducescatter_comm_op,
            allgather_comm_op_2,
            allgather_comm_op_3,
            allreduce_comm_op_2,
        ]
        return [
            attention_op,
            linear_proj_op,
            comm_ops
        ]

    
    def get_overlap_windows(self):
        overlap_windows = [(0, 2), (0, 2), (0, 0), (0, 0), (0, 1), (0, 1)]
        return overlap_windows
    
    def test_backward_config(self, monitor, test_tensors, grad_tensors, attention_fuser, overlap_windows, sm_configs):
        query_1, query_2, allgather_key, allgather_value, allreduce_inputs = test_tensors
        output_grad_1, output_grad_2, bias_grad_1, bias_grad_2 = grad_tensors
        print(f"overlap_windows: {overlap_windows}")
        print(f"sm_configs: {sm_configs}")

        t_results_list = []
        e_results_list = []
        ranks_energy_list = []

        # Warmup
        torch.cuda.profiler.start()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.profiler.start()
        for i in range(10):
            if i == 2:
                time_start = time.time()
            
            # if i == 0:
            out_1, out_2, bias_1, bias_2 = attention_fuser(
                query_1=query_1,
                query_2=query_2,
                comm_key=allgather_key,
                comm_value=allgather_value,
                comm_overlap_window_ao_ag=overlap_windows[0],
                comm_sm_configs_ao_ag=sm_configs[0],
                comm_overlap_window_ao_ar=overlap_windows[1],
                comm_sm_configs_ao_ar=sm_configs[1],
                comm_overlap_window_a_rs=overlap_windows[2],
                comm_sm_configs_a_rs=sm_configs[2],
                comm_overlap_window_a_ag=overlap_windows[3],
                comm_sm_configs_a_ag=sm_configs[3],
                comm_overlap_window_o_ag=overlap_windows[4],
                comm_sm_configs_o_ag=sm_configs[4],
                comm_overlap_window_o_ar=overlap_windows[5],
                comm_sm_configs_o_ar=sm_configs[5],
            )
            
            torch.autograd.backward(
                tensors=[out_1, out_2],
                grad_tensors=[output_grad_1, output_grad_2],
                retain_graph=True,
            )
            
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8
        torch.cuda.profiler.stop()

        # if self.rank == 0:
        #     iterations = int(8 / duration)
        #     dist_list = [iterations]
        # else:
        #     dist_list = [None]
        # dist.broadcast_object_list(dist_list, src=0, group=self.cp_group)
        # iterations = dist_list[0]
        # print(f"Duration: {duration * 1000} ms, Required iterations: {iterations}")

        # for repeat in range(self.repeat_num):
        #     torch.cuda.synchronize()
        #     dist.barrier()
        #     if self.rank == 0:
        #         monitor.begin_window("step")

        #     for i in range(iterations):
        #         torch.autograd.backward(
        #             tensors=[query, key, value, residual_out],
        #             grad_tensors=[query_grad, key_grad, value_grad, residual_grad],
        #             retain_graph=True,
        #         )
        #     torch.cuda.synchronize()
        #     dist.barrier()

        #     if self.rank == 0:
        #         result = monitor.end_window("step")
        #         t_result = result.time / iterations
        #         e_result = result.total_energy / iterations
        #         ranks_energy = [result.gpu_energy[i] / iterations for i in range(self.world_size)]
        #         t_results_list.append(t_result)
        #         e_results_list.append(e_result)
        #         ranks_energy_list.append(ranks_energy)
        
        # if self.rank == 0:
        #     with open(f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/backward_results.csv", "a") as f:
        #         line_str = f"{overlap_window[0]},{overlap_window[1]},{sm_configs[0]},{sm_configs[1]},"
        #         for i in range(self.repeat_num):
        #             line_str += f"{t_results_list[i]},{e_results_list[i]},{','.join(map(str, ranks_energy_list[i]))},"
        #         f.write(line_str.rstrip(",") + "\n")
            
    
    def run_overlap_test(self):
        test_tensors = self.create_test_tensors()
        grad_tensors = self.create_gradient_tensors()
        operations = self.create_operations(test_tensors[-1])
        comp_ops = operations[:-1]
        comm_ops = operations[-1]
        comm_ops_fwd = comm_ops[:2]
        comm_ops_bwd = comm_ops[2:]

        attention_fuser = PartitionFuser(
            ops=comp_ops,
            comm_ops_fwd=comm_ops_fwd,
            comm_ops_bwd=comm_ops_bwd,
            fuse_ops=False,
        )
        print(f"attention_fuser._forward_ops: {attention_fuser._forward_ops}")
        print(f"attention_fuser._backward_ops: {attention_fuser._backward_ops}")

        monitor = None
        # if self.rank == 0:
        #     gpu_indices = list(range(self.world_size))
        #     monitor = ZeusMonitor(gpu_indices=gpu_indices)
        #     os.makedirs(f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}", exist_ok=True)
        #     with open(f"logs/{self.model_name}/cp{self.context_parallel_size}-tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/backward_results.csv", "w") as f:
        #         title = "overlap_start,overlap_end,comm_sm_number,comm_block_size,"
        #         for i in range(self.repeat_num):
        #             title += f"{i}:time (s),{i}:total energy (J),{i}:rank0 energy (J),{i}:rank1 energy (J),"
        #         title = title.rstrip(",")
        #         title += "\n"
        #         f.write(title)
        
        # skip = True
        overlap_windows = self.get_overlap_windows()
        sm_configs = [(6, 1024), (6, 1024), (12, 1024), (6, 1024), (6, 1024), (6, 1024)]
        self.test_backward_config(
            monitor, 
            test_tensors, grad_tensors, attention_fuser, 
            overlap_windows, sm_configs
        )


def overlap_test(rank, world_size, args, master_port):
    """Run the attention fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = AttentionFuserTest(args, rank, world_size)
    try:
        test_runner.run_overlap_test()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # if rank == 0:
        #     pid = os.getpid()
        #     print(f"Killing process group {pid}")
        #     os.system(f'pkill -P {pid}')

        if dist.is_initialized():
            dist.destroy_process_group()
            print("Destroyed process group")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default=FuserTestConfig.MODEL_NAME)
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    args = parser.parse_args()

    print("Running overlap test for attention fuser")
    print(f"World size: {args.world_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Frequency: {args.frequency}")

    from torch.multiprocessing import spawn
    import random
    spawn(
        overlap_test,
        args=(
            args.world_size,
            args,
            random.randint(8000, 65535),
        ),
        nprocs=args.world_size,
        join=True,
    )
    