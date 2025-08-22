import warnings
from typing import Any, Callable, Optional

import torch
from torch.nn.parameter import Parameter

# from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.parallel_state import (
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    get_expert_tensor_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import is_te_min_version

from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs, condition_init_method
from megatron.core.num_microbatches_calculator import get_micro_batch_size

from kareus.transformer_engine.pytorch.ops import Linear


def _get_cuda_rng_tracker_fn():
    """Get the CUDA RNG tracker function for weight initialization."""
    if get_cuda_rng_tracker().is_initialized():
        return lambda: get_cuda_rng_tracker()
    return None


class TEFusibleLinear(Linear):
    """
    Wrapper for the Transformer-Engine's FusedOperation-based `Linear` layer.
    
    This is a drop-in replacement for the traditional TELinear that uses
    the experimental FusedOperation API instead of TransformerEngineBaseModule.
    
    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension
        - "row": Split the weight matrix along input dimension  
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
    ):
        self.config = config
        self.parallel_mode = parallel_mode

        # TE FusedOperation returns bias as separate output when bias=True
        # We prefer None when skip_bias_add=False, so we handle this differently
        self.te_return_bias = skip_bias_add and bias

        # FusibleLinear does not support skipping gradient accumulation for first microbatch
        # self.is_first_microbatch = True
        # self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        
        if skip_weight_param_allocation:
            raise ValueError(
                'Transformer Engine linear layers do not support skip_weight_param_allocation'
            )
        
        # FusibleLinear does not support communication overlap and rng_tracker_name
        # extra_kwargs = _get_extra_te_kwargs(config)
        # if is_te_min_version("0.8.0"):
        #     if self.config.tp_comm_overlap:
        #         if is_te_min_version("1.5.0"):
        #             # Use old overlap flags if they were supplied instead
        #             extra_kwargs["ub_overlap_ag"] = (
        #                 self.config.tp_comm_overlap_ag
        #                 if hasattr(self.config, "tp_comm_overlap_ag")
        #                 else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
        #             )
        #             extra_kwargs["ub_overlap_rs"] = (
        #                 self.config.tp_comm_overlap_rs
        #                 if hasattr(self.config, "tp_comm_overlap_rs")
        #                 else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
        #             )
        #             # Disable ub overlap for experts.
        #             if is_expert:
        #                 extra_kwargs["ub_overlap_ag"] = False
        #                 extra_kwargs["ub_overlap_rs"] = False
        #         else:
        #             extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
        #             extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
        #             extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
        #             extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
        #             # Disable ub overlap for experts.
        #             if is_expert:
        #                 extra_kwargs["ub_split_ag"] = False
        #                 extra_kwargs["ub_atomic_gemm_ag"] = False
        #                 extra_kwargs["ub_split_rs"] = False
        #                 extra_kwargs["ub_atomic_gemm_rs"] = False
        #         if is_te_min_version("1.0.0", check_equality=False):
        #             assert (
        #                 tp_comm_buffer_name is not None
        #             ), "Buffer name should be set to configure communication overlap settings"
        #             extra_kwargs["ub_name"] = tp_comm_buffer_name

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        # if is_expert:
        #     rng_tracker_name = get_expert_parallel_rng_tracker_name()
        # else:
        #     if parallel_mode == "duplicated":
        #         rng_tracker_name = get_data_parallel_rng_tracker_name()
        #     else:
        #         rng_tracker_name = None
        # if is_te_min_version("1.7.0"):
        #     extra_kwargs["rng_tracker_name"] = rng_tracker_name
        
        te_parallel_mode = parallel_mode
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group = None
            tp_size = 1
            explicit_expert_comm = False
            te_parallel_mode = None
        else:
            # Get tensor parallel group and size
            if is_expert:
                tp_group = get_expert_tensor_parallel_group(check_initialized=False)
                tp_size = get_expert_tensor_parallel_world_size()
            else:
                tp_group = get_tensor_model_parallel_group(check_initialized=False)
                tp_size = get_tensor_model_parallel_world_size()
            
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                te_parallel_mode = None
                tp_size = 1
                tp_group = None
        # Get RNG tracker function for weight initialization
        rng_tracker_fn = None
        if not is_expert:
            if parallel_mode != "duplicated":
                rng_tracker_fn = _get_cuda_rng_tracker_fn()
        
        if parallel_mode == "row":
            use_persistent_output = True
        else:
            use_persistent_output = False

        # Initialize the FusedOperation-based Linear layer
        super().__init__(
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            return_bias=self.te_return_bias,
            device=torch.cuda.current_device() if not config.use_cpu_initialization else 'cpu',
            dtype=config.params_dtype,
            tensor_parallel_mode=te_parallel_mode,
            tensor_parallel_group=tp_group,
            tensor_parallel_size=tp_size,
            sequence_parallel=config.sequence_parallel,
            rng_state_tracker_function=rng_tracker_fn,
            accumulate_into_main_grad=False,  # Let Megatron handle gradient accumulation
            use_persistent_output=use_persistent_output,
            num_batches=2,  # 2 nanobatches per microbatch
            batch_size=get_micro_batch_size() // 2, # nanobatch size
            seq_length=config.max_sequence_length,
        )

        # Handle CPU initialization if needed
        if config.use_cpu_initialization:
            self._handle_cpu_initialization(
                input_size, output_size, parallel_mode, init_method, bias, is_expert
            )

        # Set gradient reduction attributes
        self._set_gradient_attributes(parallel_mode, is_expert)

    def _handle_cpu_initialization(self, input_size, output_size, parallel_mode, init_method, bias, is_expert):
        """Handle CPU initialization of weights."""
        if is_expert:
            world_size = get_expert_tensor_parallel_world_size()
            rank = get_expert_tensor_parallel_rank()
        else:
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()

        if parallel_mode == "column":
            output_size_per_partition = divide(output_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(self.config, init_method),
                stride=1,
                return_master_weight=False,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=self.config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
        elif parallel_mode == "row":
            input_size_per_partition = divide(input_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                input_size_per_partition,
                1,
                init_method=condition_init_method(self.config, init_method),
                stride=1,
                return_master_weight=False,
                params_dtype=self.config.params_dtype,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(torch.empty(output_size, dtype=self.config.params_dtype))
                with torch.no_grad():
                    self.bias.zero_()

    def _set_gradient_attributes(self, parallel_mode, is_expert):
        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, 'allreduce', not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, 'allreduce', True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, 'sequence_parallel', self.config.sequence_parallel)

    def forward(self, x, batch_idx=0):
        """Forward pass."""
        # Call the FusedOperation forward
        outputs = super().forward(
            x,
            basic_op_kwargs=[{"batch_idx": batch_idx}, {}],
        )
        
        # Handle bias return logic to match TELinear behavior
        # if self.te_return_bias:
        #     # For FusedOperation, bias is integrated into the output
        #     # We need to return (output, bias) for compatibility
        #     return output, self.bias
        # return output, None
        assert len(outputs) == 2
        return outputs

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP for checkpointing."""
        # Provide the dist-ckpt support when TEFusibleLinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None or self.parallel_mode == "duplicated"
        ), "TEFusibleLinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, None, sharded_offsets)


class TEFusibleColumnParallelLinear(TEFusibleLinear):
    """
    Wrapper for the FusedOperation-based `Linear` layer specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
    ):
        if gather_output:
            raise ValueError('FusedOperation linear layers do not support gather_output = True')

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded."""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.weight.shape[1]}, "
            f"out_features={self.weight.shape[0]}, bias={self.bias is not None})"
        )


class TEFusibleRowParallelLinear(TEFusibleLinear):
    """
    Wrapper for the FusedOperation-based `Linear` layer specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
    ):
        if not input_is_parallel:
            raise ValueError(
                "FusedOperation linear layers do not support input_is_parallel = False"
            )

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded."""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'weight': 1}, sharded_offsets
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(in_features={self.weight.shape[1]}, "
            f"out_features={self.weight.shape[0]}, bias={self.bias is not None})"
        )


# Example usage and compatibility check
def check_te_fused_operation_availability():
    """Check if TransformerEngine FusedOperation Linear is available."""
    try:
        from transformer_engine.pytorch.ops.linear import Linear as TEFusibleLinear
        return True
    except ImportError:
        return False


def get_fused_linear_class():
    """Get the appropriate Linear class based on availability."""
    if check_te_fused_operation_availability():
        warnings.warn(
            "Using experimental FusedOperation-based Linear implementation. "
            "This API is subject to change and may not have all features of the stable API.",
            UserWarning
        )
        return TEFusibleLinear
    else:
        # Fallback to traditional implementation
        from megatron.core.extensions.transformer_engine import TELinear
        return TELinear


# Drop-in replacement functions for easy migration
def create_te_linear_layer(
    input_size: int,
    output_size: int,
    *,
    parallel_mode: Optional[str],
    config: TransformerConfig,
    init_method: Callable,
    bias: bool,
    skip_bias_add: bool,
    skip_weight_param_allocation: bool,
    tp_comm_buffer_name: Optional[str] = None,
    is_expert: bool = False,
    use_fused_operation: bool = True,
):
    """Create a linear layer using either FusedOperation or traditional approach."""
    if use_fused_operation and check_te_fused_operation_availability():
        return TEFusibleLinear(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        )
    else:
        # Fallback to traditional TELinear
        from megatron.core.extensions.transformer_engine import TELinear
        return TELinear(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
        ) 