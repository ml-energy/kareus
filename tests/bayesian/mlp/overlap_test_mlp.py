import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../fuser/'))

from megatron.core.transformer.transformer_config import TransformerConfig
from common_config import FuserTestConfig
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from kareus.megatron.core.extensions.ops import BiasSwigluOp
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.transformer_engine.pytorch.ops.linear import Linear
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser
from zeus.monitor import ZeusMonitor
# from cfuser.core.utils import nvtx_range
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
    tp_group = dist.new_group(ranks)
    print(f"Created tensor parallel group with ranks: {ranks}")
    return tp_group


class MLPFuserTest:
    """Test suite for MLP fuser operations."""

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        assert self.world_size == args.tensor_parallel_size
        self.context_parallel_size = args.context_parallel_size
        self.tensor_parallel_size = args.tensor_parallel_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(rank, world_size)
        
        # Test configuration
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len // args.context_parallel_size
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE
        
        # Create transformer config
        self.config = FuserTestConfig.create_mlp_config(
            context_parallel_size=1,
            tensor_parallel_size=world_size, 
            dtype=self.dtype
        )

        self.frequency = args.frequency if hasattr(args, "frequency") else None
        self.repeat_num = 1
    
    def create_test_tensors(self):
        """Create test tensors for the MLP operations."""
        nano_batch_size = self.batch_size // 2
        hidden_states = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        # bias = torch.randn(
        #     self.hidden_size,
        #     dtype=self.dtype, device=self.device, requires_grad=True
        # )
        bias = None
        residual = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )

        allreduce_inputs = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=True
        )
        
        return hidden_states, bias, residual, allreduce_inputs
    
    def create_operations(self, allreduce_inputs):
        """Create all the required operations for the MLP fuser."""
        
        # 1. BDA Operation (Bias Dropout Add)
        bda_op = BiasDropoutAddOp(
            dropout_prob=self.config.hidden_dropout,
            training=True
        )
        
        # 2. LayerNorm Operation
        layernorm_op = RMSNorm(
            normalized_shape=self.hidden_size,
            eps=self.config.layernorm_epsilon,
            device=self.device,
            dtype=self.dtype
        )
        
        # 3. Linear FC1 Operation (input to intermediate with gating)
        # Since gated_linear_unit=True, output is 2 * ffn_hidden_size
        fc1_hidden_size = 2 * self.ffn_hidden_size // self.tensor_parallel_size
        linear_fc1_op = Linear(
            in_features=self.hidden_size,
            out_features=fc1_hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=False,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )

        # 4. BiasSwigluOp (activation function with bias)
        bias_swiglu_op = BiasSwigluOp(
            fp8_input_store=self.config.activation_func_fp8_input_store
        )

        # 5. Linear FC2 Operation (intermediate back to hidden)
        fc2_hidden_size = self.ffn_hidden_size // self.tensor_parallel_size
        linear_fc2_op = Linear(
            in_features=fc2_hidden_size,
            out_features=self.hidden_size,
            device=self.device,
            dtype=self.dtype,
            bias=False,
            return_bias=True,
            tensor_parallel_mode=None,
            tensor_parallel_group=None,
            tensor_parallel_size=None,
        )

        # 6. AllReduce Communication Operation
        if self.tensor_parallel_size > 1:
            nano_batch_size = self.batch_size // 2
            allreduce_comm_op = AllReduce(
                process_group=self.tp_group,
                async_op=True,
                backend="msccl",
                rank=self.rank,
                world_size=self.world_size,
                use_persistent_output=True,
                input_buffer=allreduce_inputs,
                tensor_size=[self.seq_length, nano_batch_size, self.hidden_size],
                device=self.device,
                dtype=self.dtype,
            )

            return [
                bda_op,
                layernorm_op,
                linear_fc1_op,
                bias_swiglu_op,
                linear_fc2_op,
                allreduce_comm_op,
            ]
        else:
            raise ValueError("Tensor parallel size must be greater than 1")

    
    def get_overlap_windows(self):
        overlap_windows = [
            (-1, -1),
            (0, 1), (2, 3), (4, 4), (5, 6),
            (0, 3), (2, 4), (4, 6),
            (0, 4), (2, 6),
            (0, 6),
            (0, 6),
        ]
        return overlap_windows
    
    def test_config(self, monitor, test_tensors, mlp_fuser, overlap_window, sm_configs):
        hidden_states, bias, residual, allreduce_inputs = test_tensors
        t_results_list = []
        e_results_list = []
        ranks_energy_list = []
        # if self.rank == 0:
        #     os.makedirs(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/" \
        #         f"{self.frequency}/overlap_{overlap_window[0]}_{overlap_window[1]}_sm_{sm_configs[0]}_{sm_configs[1]}", exist_ok=True)

        # Warmup
        torch.cuda.synchronize()
        dist.barrier()
        for i in range(10):
            if i == 2:
                time_start = time.time()
            output, output_bias, output_residual, allreduce_output = mlp_fuser(
                hidden_states=hidden_states,
                bias=bias,
                residual=residual,
                comm_input=allreduce_inputs,
                comm_overlap_window=overlap_window,
                comm_sm_configs=sm_configs,
            )
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8
        if self.rank == 0:
            iterations = int(8 / duration)
            dist_list = [iterations]
        else:
            dist_list = [None]
        dist.broadcast_object_list(dist_list, src=0, group=self.tp_group)
        iterations = dist_list[0]
        print(f"Duration: {duration * 1000} ms, Required iterations: {iterations}")

        for repeat in range(self.repeat_num):
            # manager = multiprocessing.Manager()
            # temperature_data = manager.list()
            # proc = temperature_start(temperature_data, self.rank)

            torch.cuda.synchronize()
            dist.barrier()
            if self.rank == 0:
                monitor.begin_window("step")

            for i in range(iterations):
                output, output_bias, output_residual, allreduce_output = mlp_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    comm_input=allreduce_inputs,
                    comm_overlap_window=overlap_window,
                    comm_sm_configs=sm_configs,
                )
            torch.cuda.synchronize()
            dist.barrier()

            if self.rank == 0:
                result = monitor.end_window("step")
                t_result = result.time / iterations
                e_result = result.total_energy / iterations
                ranks_energy = [result.gpu_energy[i] / iterations for i in range(self.world_size)]
                t_results_list.append(t_result)
                e_results_list.append(e_result)
                ranks_energy_list.append(ranks_energy)

            # temperature_end(proc, temperature_data)
            # with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/" \
            #     f"{self.frequency}/overlap_{overlap_window[0]}_{overlap_window[1]}_sm_{sm_configs[0]}_{sm_configs[1]}/" \
            #     f"gpu{self.rank}_iter{repeat}.csv", "w") as f:
                
            #     file_str = "timestamp,temperature,clock,power\n"
            #     for data in temperature_data:
            #         file_str += ",".join(map(str, data)) + "\n"
            #     f.write(file_str)
        
        if self.rank == 0:
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/energy_results.csv", "a") as f:
                line_str = f"{overlap_window[0]},{overlap_window[1]},{sm_configs[0]},{sm_configs[1]},"
                for i in range(self.repeat_num):
                    line_str += f"{t_results_list[i]},{e_results_list[i]},{','.join(map(str, ranks_energy_list[i]))},"
                f.write(line_str.rstrip(",") + "\n")
            
    
    def run_overlap_test(self):
        test_tensors = self.create_test_tensors()
        operations = self.create_operations(test_tensors[-1])
        comp_ops = operations[:-1]
        allreduce_comm_op = operations[-1]

        mlp_fuser = PartitionFuser(
            ops=comp_ops,
            comm_op_fwd=allreduce_comm_op,
            fuse_ops=False
        )
        print(f"mlp_fuser._forward_ops: {mlp_fuser._forward_ops}")
        print(f"mlp_fuser._backward_ops: {mlp_fuser._backward_ops}")

        monitor = None
        if self.rank == 0:
            gpu_indices = list(range(self.world_size))
            monitor = ZeusMonitor(gpu_indices=gpu_indices)
            os.makedirs(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}", exist_ok=True)
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/energy_results.csv", "w") as f:
                title = "overlap_start,overlap_end,comm_sm_number,comm_block_size,"
                for i in range(self.repeat_num):
                    title += f"{i}:time (s),{i}:total energy (J),{i}:rank0 energy (J),{i}:rank1 energy (J),"
                title = title.rstrip(",")
                title += "\n"
                f.write(title)
        
        # skip = True
        overlap_windows = self.get_overlap_windows()
        for overlap_window in overlap_windows:
            # if overlap_window[0] == 2 and overlap_window[1] == 2:
            #     skip = False
            # if skip:
            #     continue
            for sm_num in range(1, 21):
                for block_size in [512, 1024]:
                    # if sm_num == 17 and block_size == 512 and overlap_window[0] == 4 and overlap_window[1] == 5:
                    #     skip = False
                    # if skip:
                    #     continue
                    sm_configs = (sm_num, block_size)
                    print(f"Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}")
                    # with nvtx_range(f"Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}"):
                    self.test_config(
                        monitor, 
                        test_tensors, mlp_fuser, 
                        overlap_window, sm_configs
                    )
                    # return
                    # time.sleep(30)


def overlap_test(rank, world_size, args, master_port):
    """Run the MLP fuser tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run tests
    test_runner = MLPFuserTest(args, rank, world_size)
    try:
        test_runner.run_overlap_test()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if rank == 0:
            pid = os.getpid()
            print(f"Killing process group {pid}")
            os.system(f'pkill -P {pid}')

        if dist.is_initialized():
            dist.destroy_process_group()
            print("Destroyed process group")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=FuserTestConfig.DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--context_parallel_size", "-cp", type=int, default=FuserTestConfig.DEFAULT_CONTEXT_PARALLEL_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    args = parser.parse_args()

    print("Running overlap test for MLP fuser")
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
