import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from kareus.megatron.core.extensions.qkv_postprocess_op import QKVPostProcessOp
from kareus.megatron.core.extensions.bias_swiglu_op import BiasSwigluOp
from kareus.megatron.core.extensions.rotary_embedding_op import RotaryEmbeddingOp
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.te_attention import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
from kareus.megatron.core.extensions.attention_fuser import AttentionFuser
from kareus.megatron.core.extensions.partition_fuser_profile import PartitionFuser
from megatron.core.transformer.enums import AttnMaskType
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


def gpu_temperature_monitor(shared_list, rank):
    """
    Child process function that continuously collects GPU temperature data.
    The data is stored in the 'shared_list' as [timestamp, temperature].
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)

    try:
        while True:
            ts = time.time()
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            power = int(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
            shared_list.append([ts, temp, sm_clock, power])
            # time.sleep(interval)
    except:
        pass
    finally:
        pynvml.nvmlShutdown()


def temperature_start(temperature_data, rank):
    p = multiprocessing.Process(
        target=gpu_temperature_monitor, 
        args=(temperature_data, rank)
    )
    p.start()
    return p


def temperature_end(p, temperature_data):
    print("Terminating temperature monitoring process")
    p.terminate()
    try:
        p.join(timeout=5)
        print("Temperature monitoring process joined")
    except Exception as e:
        print(f"Error joining temperature monitoring process: {e}")
    collected_data = list(temperature_data)

    filtered_data = []
    previous_temp = None
    previous_clock = None
    previous_power = None
    for ts, temp, sm_clock, power in collected_data:
        if temp != previous_temp or sm_clock != previous_clock or power != previous_power:
            filtered_data.append([ts, temp, sm_clock, power])
        previous_temp = temp
        previous_clock = sm_clock
        previous_power = power
    return filtered_data


class MLPFuserTest:
    """Test suite for attention fuser operations."""

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(rank, world_size)
        
        # Test configuration
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = FuserTestConfig.HIDDEN_SIZE
        self.num_attention_heads = FuserTestConfig.NUM_ATTENTION_HEADS
        self.num_query_groups = FuserTestConfig.NUM_QUERY_GROUPS
        self.ffn_hidden_size = FuserTestConfig.FFN_HIDDEN_SIZE
        
        # Create transformer config
        self.config = FuserTestConfig.create_mlp_config(world_size, self.dtype)

        self.frequency = args.frequency
        self.repeat_num = 1
    
    def create_test_tensors(self):
        """Create test tensors for the attention operations."""
        nano_batch_size = self.batch_size
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
    
    def create_operations(self):
        """Create all the required operations for the attention fuser."""
        
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
                linear_fc1_op,
                bias_swiglu_op,
                linear_fc2_op,
                allreduce_comm_op,
            ]

        else:
            raise ValueError("Tensor parallel size must be greater than 1")
    
    # def get_overlap_windows(self):
    #     overlap_windows = [
    #         (-1, -1),
    #         (0, 1), (2, 3), (4, 5), (6, 6), (7, 8),
    #         (0, 3), (2, 5), (4, 6), (6, 8),
    #         (0, 5), (2, 6), (4, 8),
    #         (0, 6), (2, 8),
    #         (0, 8),
    #     ]
    #     return overlap_windows
    
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
                allreduce_input=allreduce_inputs,
                allreduce_overlap_window=overlap_window,
                allreduce_sm_configs=sm_configs,
            )
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8
        if self.rank == 0:
            iterations = int(10 / duration)
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
                    allreduce_input=allreduce_inputs,
                    allreduce_overlap_window=overlap_window,
                    allreduce_sm_configs=sm_configs,
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
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/energy_results_baseline.csv", "a") as f:
                line_str = f"{overlap_window[0]},{overlap_window[1]},{sm_configs[0]},{sm_configs[1]},"
                for i in range(self.repeat_num):
                    line_str += f"{t_results_list[i]},{e_results_list[i]},{','.join(map(str, ranks_energy_list[i]))},"
                f.write(line_str.rstrip(",") + "\n")
            
    
    def run_overlap_test(self):
        test_tensors = self.create_test_tensors()
        operations = self.create_operations()
        comp_ops = operations[:-1]
        allreduce_comm_op = operations[-1]

        mlp_fuser = PartitionFuser(
            ops=comp_ops,
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=False
        )
        print(f"mlp_fuser._forward_ops: {mlp_fuser._forward_ops}")
        print(f"mlp_fuser._backward_ops: {mlp_fuser._backward_ops}")

        monitor = None
        if self.rank == 0:
            gpu_indices = list(range(self.world_size))
            monitor = ZeusMonitor(gpu_indices=gpu_indices)
            os.makedirs(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}", exist_ok=True)
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/energy_results_baseline.csv", "w") as f:
                title = "overlap_start,overlap_end,comm_sm_number,comm_block_size,"
                for i in range(self.repeat_num):
                    title += f"{i}:time (s),{i}:total energy (J),{i}:rank0 energy (J),{i}:rank1 energy (J),"
                title = title.rstrip(",")
                title += "\n"
                f.write(title)
        
        overlap_window = (-1, -1)
        sm_num, block_size = None, None
        sm_configs = (sm_num, block_size)
        print(f"Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}")
        # with nvtx_range(f"Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}"):
        self.test_config(
            monitor, 
            test_tensors, mlp_fuser, 
            overlap_window, sm_configs
        )


def overlap_test(rank, world_size, args, master_port):
    """Run the attention fuser tests in a distributed environment."""
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
    parser.add_argument("--world_size", "-w", type=int, default=FuserTestConfig.DEFAULT_WORLD_SIZE)
    parser.add_argument("--batch_size", "-b", type=int, default=FuserTestConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq_len", "-s", type=int, default=FuserTestConfig.DEFAULT_SEQ_LENGTH)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    args = parser.parse_args()

    print("Running overlap test for mlp fuser")
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
    