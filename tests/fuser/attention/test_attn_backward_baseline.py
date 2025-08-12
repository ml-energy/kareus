import os
import torch
import torch.distributed as dist
import time
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from megatron.core.transformer.transformer_config import TransformerConfig
from kareus.transformer_engine.pytorch.ops.basic.bias_dropout_add import BiasDropoutAddOp
from kareus.transformer_engine.pytorch.ops.basic.layer_norm import LayerNorm
from kareus.transformer_engine.pytorch.ops.basic.rmsnorm import RMSNorm
from kareus.megatron.core.extensions.qkv_postprocess_op import QKVPostProcessOp
from kareus.megatron.core.extensions.rotary_embedding_op import RotaryEmbeddingOp
from kareus.transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from kareus.megatron.core.extensions.te_attention import TEFusibleDotProductAttention
from kareus.transformer_engine.pytorch.ops.linear import Linear
from kareus.megatron.core.extensions.attention_fuser import AttentionFuser
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser
from megatron.core.transformer.enums import AttnMaskType
from zeus.monitor import ZeusMonitor
from cfuser.core.utils import nvtx_range
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


class AttentionFuserBackwardTest:
    """Test suite for attention fuser backward pass operations."""

    def __init__(self, args, rank: int = 0, world_size: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.rank = rank
        self.world_size = world_size
        self.tensor_parallel_size = world_size
        
        # Initialize distributed processing
        self.tp_group = init_distributed(rank, world_size)
        
        # Test configuration
        self.batch_size = args.batch_size
        self.seq_length = args.seq_len
        self.hidden_size = 3072
        self.num_attention_heads = 24
        self.num_query_groups = 8  # For grouped query attention
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.ffn_hidden_size = 8192
        
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
            add_bias_linear=False,
        )

        self.frequency = args.frequency
        self.repeat_num = 1
        self.overlap_window_forward = (-1, -1)
        self.sm_configs_forward = (20, 1024)
    
    def create_test_tensors(self):
        """Create test tensors for the attention operations with gradients enabled."""
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
    
    def create_gradient_tensors(self):
        """Create gradient tensors for backward pass testing."""
        nano_batch_size = self.batch_size
        
        # Gradient for main output
        output_grad = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )
        
        # Gradient for bias output
        # bias_grad = torch.randn(
        #     self.hidden_size,
        #     dtype=self.dtype, device=self.device
        # )
        bias_grad = None
        
        # Gradient for residual output
        residual_grad = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )
        
        # Gradient for allreduce input
        allreduce_input_grad = torch.randn(
            self.seq_length, nano_batch_size, self.hidden_size,
            dtype=self.dtype, device=self.device
        )
        
        return output_grad, bias_grad, residual_grad, allreduce_input_grad
    
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
            bias=False,
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
                linear_qkv_op,
                qkv_postprocess_op,
                rotary_embedding_op,
                attention_op,
                linear_proj_op,
                allreduce_comm_op
            ]

        else:
            raise ValueError("Tensor parallel size must be greater than 1")
    
    # def get_backward_overlap_windows(self):
    #     overlap_windows = [
    #         (-1, -1),
    #         (0, 1), (2, 2), (3, 4), (5, 6), (7, 8),
    #         (0, 2), (2, 4), (3, 6), (5, 8),
    #         (0, 4), (2, 6), (3, 8),
    #         (0, 6), (2, 8),
    #         (0, 8),
    #     ]
    #     return overlap_windows
    
    def test_backward_config(self, monitor, test_tensors, grad_tensors, attention_fuser, overlap_window, sm_configs):
        """Test backward pass configuration with specific overlap window and SM configs."""
        hidden_states, bias, residual, rotary_pos_emb, attention_mask, allreduce_inputs = test_tensors
        output_grad, bias_grad, residual_grad, allreduce_input_grad = grad_tensors
        
        t_results_list = []
        e_results_list = []
        ranks_energy_list = []

        # Warmup for backward pass
        torch.cuda.synchronize()
        dist.barrier()
        for i in range(10):
            if i == 2:
                time_start = time.time()

            # Clear gradients
            # if hidden_states.grad is not None:
            #     hidden_states.grad.zero_()
            # if bias.grad is not None:
            #     bias.grad.zero_()
            # if residual.grad is not None:
            #     residual.grad.zero_()
            # if allreduce_inputs.grad is not None:
            #     allreduce_inputs.grad.zero_()
                
            # Forward pass
            if i == 0:
                output, output_bias, output_residual, allreduce_output = attention_fuser(
                    hidden_states=hidden_states,
                    bias=bias,
                    residual=residual,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_mask=attention_mask,
                    allreduce_input=allreduce_inputs,
                    allreduce_overlap_window=self.overlap_window_forward,
                    allreduce_sm_configs=self.sm_configs_forward,
                    allreduce_overlap_window_backward=overlap_window,
                    allreduce_sm_configs_backward=sm_configs,
                )
            
            # Backward pass - this is what we're testing
            torch.autograd.backward(
                tensors=[output, output_residual, allreduce_output],
                grad_tensors=[output_grad, residual_grad, allreduce_input_grad],
                retain_graph=True,
            )
            
        torch.cuda.synchronize()
        dist.barrier()
        time_end = time.time()
        duration = (time_end - time_start) / 8  # 8 iterations after warmup
        
        if self.rank == 0:
            iterations = int(10 / duration)
            dist_list = [iterations]
        else:
            dist_list = [None]
        dist.broadcast_object_list(dist_list, src=0, group=self.tp_group)
        iterations = dist_list[0]
        if iterations is None:
            iterations = 10  # fallback value
        print(f"Total Duration: {duration * 1000} ms, Required iterations: {iterations}")

        for repeat in range(self.repeat_num):
            torch.cuda.synchronize()
            dist.barrier()
            if self.rank == 0:
                monitor.begin_window("step")

            for i in range(iterations):
                # # Clear gradients
                # if hidden_states.grad is not None:
                #     hidden_states.grad.zero_()
                # if bias.grad is not None:
                #     bias.grad.zero_()
                # if residual.grad is not None:
                #     residual.grad.zero_()
                # if allreduce_inputs.grad is not None:
                #     allreduce_inputs.grad.zero_()
                
                # # Forward pass
                # output, output_bias, output_residual, grad_allreduce_input = attention_fuser(
                #     hidden_states=hidden_states,
                #     bias=bias,
                #     residual=residual,
                #     rotary_pos_emb=rotary_pos_emb,
                #     attention_mask=attention_mask,
                #     allreduce_input=allreduce_inputs,
                #     allreduce_overlap_window=self.overlap_window_forward,
                #     allreduce_sm_configs=self.sm_configs_forward,
                #     allreduce_overlap_window_backward=overlap_window,
                #     allreduce_sm_configs_backward=sm_configs,
                # )
                
                # Backward pass - this is what we're measuring
                torch.autograd.backward(
                    tensors=[output, output_residual, allreduce_output],
                    grad_tensors=[output_grad, residual_grad, allreduce_input_grad],
                    retain_graph=True,
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
        
        if self.rank == 0:
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/backward_energy_results_baseline.csv", "a") as f:
                line_str = f"{overlap_window[0]},{overlap_window[1]},{sm_configs[0]},{sm_configs[1]},"
                for i in range(self.repeat_num):
                    line_str += f"{t_results_list[i]},{e_results_list[i]},{','.join(map(str, ranks_energy_list[i]))},"
                f.write(line_str.rstrip(",") + "\n")
    
    def run_overlap_test(self):
        """Run backward pass overlap tests."""
        test_tensors = self.create_test_tensors()
        grad_tensors = self.create_gradient_tensors()
        operations = self.create_operations()
        comp_ops = operations[:7]
        allreduce_comm_op = operations[7]

        attention_fuser = PartitionFuser(
            ops=comp_ops,
            allreduce_comm_op=allreduce_comm_op,
            fuse_ops=False
        )
        print(f"attention_fuser._forward_ops: {attention_fuser._forward_ops}")
        print(f"attention_fuser._backward_ops: {attention_fuser._backward_ops}")

        monitor = None
        if self.rank == 0:
            gpu_indices = list(range(self.world_size))
            monitor = ZeusMonitor(gpu_indices=gpu_indices)
            os.makedirs(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}", exist_ok=True)
            with open(f"logs/tp{self.tensor_parallel_size}-bs{self.batch_size}-seq{self.seq_length}/{self.frequency}/backward_energy_results_baseline.csv", "w") as f:
                title = "overlap_start,overlap_end,comm_sm_number,comm_block_size,"
                for i in range(self.repeat_num):
                    title += f"{i}:time (s),{i}:total energy (J),{i}:rank0 energy (J),{i}:rank1 energy (J),"
                title = title.rstrip(",")
                title += "\n"
                f.write(title)
        
        overlap_window = (-1, -1)
        sm_num, block_size = None, None
        sm_configs = (sm_num, block_size)
        print(f"Backward Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}")
        with nvtx_range(f"Backward Overlap {overlap_window} - SM: {sm_num}, Block: {block_size}"):
            self.test_backward_config(
                monitor, 
                test_tensors, 
                grad_tensors,
                attention_fuser, 
                overlap_window, 
                sm_configs
            )
        # return


def overlap_test(rank, world_size, args, master_port):
    """Run the attention fuser backward pass tests in a distributed environment."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{master_port}"

    # Create test instance and run backward tests
    test_runner = AttentionFuserBackwardTest(args, rank, world_size)
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
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    args = parser.parse_args()

    print("Running backward pass overlap test for attention fuser")
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