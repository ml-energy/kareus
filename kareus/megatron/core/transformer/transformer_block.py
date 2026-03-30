"""
Modified from Megatron-LM (megatron/core/transformer/transformer_block.py) by NVIDIA.
Changes: replaced per-layer sequential forward with graph-based partition system
(TensorGraph, PartitionBuilder, TransformerBlockAutogradFunction) for automatic
tensor routing and communication-computation overlap; forward splits micro-batches
into nanobatches; communication operators (AllReduce, AllGatherKV, ReduceScatterKV)
assigned lazily via set_tensor_parallel_group/set_context_parallel_group.
"""

from typing import Optional, Union, List

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, make_viewless_tensor

from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    _get_block_submodules,
)

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.num_microbatches_calculator import get_micro_batch_size

from kareus.megatron.core.transformer.transformer_layer import TransformerLayer

from kareus.transformer_engine.pytorch.ops import AllGatherKV
from kareus.transformer_engine.pytorch.ops import ReduceScatterKV
from kareus.transformer_engine.pytorch.ops import AllReduce


from kareus.megatron.core.partitions import (
    CommunicationType,
    ComputeOp,
    PartitionBuilder,
    SeedConfig,
    TensorGraphBuilder,
    TransformerBlockAutogradFunction,
)
from kareus.megatron.core.partitions.tensor_graph import ComputeOpSpec


class TransformerBlock(MegatronModule):
    """Transformer class.

    Uses graph-based partition system for automatic tensor routing and
    communication-computation overlap with nanobatch interleaving.

    Initialization flow:
        1. __init__  -> _build_layers() -> _build_partitions()
           (CommunicationOps have operator=None at this point)
        2. set_tensor_parallel_group()  -> _assign_comm_operators(ALL_REDUCE)
        3. set_context_parallel_group() -> _assign_comm_operators(ALL_GATHER_KV, REDUCE_SCATTER_KV)
        4. forward() -> split NB -> TransformerBlockAutogradFunction -> concat -> layernorm
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config)

        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        # required for pipeline parallel schedules
        self.input_tensor = None

        if self.config.cpu_offloading:
            raise NotImplementedError(
                "CPU offloading is not supported in Kareus TransformerBlock"
            )

        if self.config.fp8:
            raise NotImplementedError(
                "FP8 training is not supported in Kareus TransformerBlock"
            )

        if self.config.sequence_parallel:
            raise NotImplementedError(
                "Sequence parallel is not supported in Kareus TransformerBlock"
            )

        if self.config.distribute_saved_activations:
            raise NotImplementedError(
                "distribute_saved_activations is not supported in Kareus TransformerBlock"
            )

        if self.config.recompute_granularity == 'selective':
            raise NotImplementedError(
                "Selective activation checkpointing is not supported in Kareus TransformerBlock"
            )

        if (
            self.config.recompute_method == 'block'
        ):
            raise NotImplementedError(
                "recompute_method='block' is not supported in Kareus TransformerBlock; "
                "only 'uniform' (checkpoint all layers at once) is supported"
            )

        self._build_layers()

        # Scheduler (from config)
        self.scheduler = getattr(self.config, 'kareus_scheduler', None)

        # Build tensor graphs and partitions
        self._build_partitions()

    def _build_partitions(self):
        """Build TensorGraphs and form interleaved nanobatch partitions.

        Called from __init__ after layers are built.
        CommunicationOps have operator=None at this point; physical comm
        operators are assigned later by set_tensor_parallel_group() /
        set_context_parallel_group().
        """
        # Step 1: Collect all operators from all layers
        all_ops = []
        for layer in self.layers:
            all_ops.extend(layer.get_all_operators())

        # Step 2: Build FORWARD TensorGraph
        fwd_builder = TensorGraphBuilder()
        fwd_builder.add_initial_channels({
            "main": "ext_main",
            "rotary_pos_emb": "ext_rotary_pos_emb",
        })

        # Track op_id mapping: id(operator) -> forward op_id
        # Needed so backward ComputeOps get matching op_ids for
        # OperationContext retrieval.
        fwd_op_id_map = {}  # id(operator_object) -> op_id
        for op in all_ops:
            for spec in op.get_forward_ops():
                concrete_op = fwd_builder.add_op(spec)
                if isinstance(concrete_op, ComputeOp):
                    fwd_op_id_map[id(spec.operator)] = concrete_op.op_id

        self.forward_tensor_graph = fwd_builder.build()

        # Step 3: Build BACKWARD TensorGraph
        bwd_builder = TensorGraphBuilder()
        bwd_builder.add_initial_channels({
            "grad_main": "ext_grad_main",
        })

        for op in reversed(all_ops):
            for spec in op.get_backward_ops():
                # Reuse forward op_id so backward can retrieve the
                # OperationContext saved during forward.
                if isinstance(spec, ComputeOpSpec) and spec.op_id is None:
                    fwd_id = fwd_op_id_map.get(id(spec.operator))
                    if fwd_id is not None:
                        spec.op_id = fwd_id
                bwd_builder.add_op(spec)

        self.backward_tensor_graph = bwd_builder.build()

        # Step 4: Form interleaved partitions
        builder = PartitionBuilder(
            forward_graph=self.forward_tensor_graph,
            backward_graph=self.backward_tensor_graph,
        )
        self.forward_partitions = builder.build_forward_partitions()
        self.backward_partitions = builder.build_backward_partitions()

        # Step 5: Create SeedConfig (defaults match the initial channels above)
        self.seed_config = SeedConfig()

    def _assign_comm_operators(self, comm_type, comm_ops):
        """Assign physical comm operators to CommunicationOps of given type.

        Each partition has its own cloned CommunicationOp instance (see
        PartitionBuilder._clone_comm), so assigning different physical
        operators to NB0 vs NB1 partitions is safe.

        Args:
            comm_type: CommunicationType enum value to match.
            comm_ops: List of 2 physical comm operators [nb0_op, nb1_op].
        """
        for partition in self.forward_partitions + self.backward_partitions:
            if (
                partition.comm_op is not None
                and partition.comm_op.comm_type == comm_type
            ):
                partition.comm_op.operator = comm_ops[partition.nano_batch_idx]

    def set_tensor_parallel_group(
        self, tp_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> None:
        """Create AllReduce operators and assign to partitions."""
        nano_batch_size = get_micro_batch_size() // 2
        local_seq_length = (
            self.config.max_sequence_length // self.config.context_parallel_size
        )
        hidden_size = self.config.hidden_size

        self.allreduce_comm_ops = []
        for i in range(2):  # two nano-batches
            allreduce_comm_op = AllReduce(
                process_group=get_tensor_model_parallel_group(
                    check_initialized=False
                ),
                async_op=True,
                backend="msccl",
                rank=get_tensor_model_parallel_rank(),
                world_size=get_tensor_model_parallel_world_size(),
                tensor_size=[local_seq_length, nano_batch_size, hidden_size],
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
                batch_idx=i,
            )
            self.allreduce_comm_ops.append(allreduce_comm_op)

        self._assign_comm_operators(
            CommunicationType.ALL_REDUCE, self.allreduce_comm_ops
        )

    def set_context_parallel_group(
        self, cp_group, cp_global_ranks, cp_stream
    ) -> None:
        """Create AllGather/ReduceScatter operators and assign to partitions."""
        nano_batch_size = get_micro_batch_size() // 2
        local_query_groups = (
            self.config.num_query_groups // self.config.tensor_model_parallel_size
        )

        self.allgather_comm_ops = []
        self.reducescatter_comm_ops = []

        for i in range(2):  # two nano-batches
            allgather_comm_op = AllGatherKV(
                process_group=get_context_parallel_group(
                    check_initialized=False
                ),
                async_op=True,
                backend="msccl",
                rank=get_context_parallel_rank(),
                world_size=get_context_parallel_world_size(),
                tensor_size=[
                    self.config.max_sequence_length,
                    nano_batch_size,
                    local_query_groups,
                    self.config.kv_channels,
                ],
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
                batch_idx=i,
            )
            self.allgather_comm_ops.append(allgather_comm_op)

        for i in range(2):
            reducescatter_comm_op = ReduceScatterKV(
                process_group=get_context_parallel_group(
                    check_initialized=False
                ),
                async_op=True,
                backend="msccl",
                rank=get_context_parallel_rank(),
                world_size=get_context_parallel_world_size(),
                tensor_size=[
                    self.config.max_sequence_length,
                    nano_batch_size,
                    local_query_groups,
                    self.config.kv_channels,
                ],
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
                batch_idx=i,
            )
            self.reducescatter_comm_ops.append(reducescatter_comm_op)

        self._assign_comm_operators(
            CommunicationType.ALL_GATHER_KV, self.allgather_comm_ops
        )
        self._assign_comm_operators(
            CommunicationType.REDUCE_SCATTER_KV, self.reducescatter_comm_ops
        )

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            fp8_init_context = get_fp8_context(layer_config, global_layer_number - 1, is_init=True)

            # Extract TransformerLayerSubmodules from ModuleSpec if needed
            if isinstance(layer_spec, ModuleSpec):
                if hasattr(layer_spec, 'submodules') and layer_spec.submodules is not None:
                    layer_submodules = layer_spec.submodules
                else:
                    raise ValueError("ModuleSpec does not contain submodules")
            else:
                layer_submodules = layer_spec

            with fp8_init_context:
                layer = TransformerLayer(
                    config=layer_config,
                    submodules=layer_submodules,
                    layer_number=layer_number,
                )
            return layer

        layers = []
        for i, layer_spec in enumerate(self.submodules.layer_specs):
            layer = build_layer(layer_spec, i + 1)
            layers.append(layer)

        self.layers = torch.nn.ModuleList(layers)

        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

    def _get_all_params(self) -> List[torch.nn.Parameter]:
        """Get all parameters in op_id order for autograd tracking.

        Iterates operators in the same order as ``get_all_operators()``
        (which matches forward graph op_id assignment order), so the
        returned list is aligned with how ``_combine_param_grads``
        accumulates backward gradients.
        """
        params = []
        for layer in self.layers:
            for op in layer.get_all_operators():
                params.extend(op.parameters())
        return params

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """Forward pass through the transformer block.

        Same external interface as the old fuser-based implementation.
        Internally uses TransformerBlockAutogradFunction for graph-based
        partition execution with automatic tensor routing.

        Args:
            hidden_states: Input tensor [s, b, h].
            attention_mask: Boolean mask [1, 1, s, s] (currently must be None).
            rotary_pos_emb: Rotary positional embeddings (shared across NB).
            ...remaining args as before...

        Returns:
            Output hidden states tensor [s, b, h].
        """
        if context is not None or context_mask is not None:
            raise NotImplementedError(
                "Cross-attention (context/context_mask) is not supported in Kareus TransformerBlock"
            )

        if rotary_pos_cos is not None or rotary_pos_sin is not None:
            raise NotImplementedError(
                "rotary_pos_cos/rotary_pos_sin is not supported in Kareus TransformerBlock; "
                "use rotary_pos_emb instead"
            )

        if attention_bias is not None:
            raise NotImplementedError(
                "attention_bias is not supported in Kareus TransformerBlock"
            )

        if packed_seq_params is not None:
            raise NotImplementedError(
                "packed_seq_params is not supported in Kareus TransformerBlock"
            )

        if sequence_len_offset is not None:
            raise NotImplementedError(
                "sequence_len_offset is not supported in Kareus TransformerBlock"
            )

        if inference_context is not None or inference_params is not None:
            raise NotImplementedError(
                "Inference is not supported in Kareus TransformerBlock"
            )

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )

        # Split into nano-batches
        batch_size = hidden_states.size(1)
        if batch_size < 2:
            raise ValueError(
                f"Batch size must be at least 2 for nano-batch splitting, got {batch_size}"
            )
        mid_point = batch_size // 2
        h1 = hidden_states[:, :mid_point, ...]
        h2 = hidden_states[:, mid_point:, ...]

        # Collect all params for autograd tracking
        all_params = self._get_all_params()

        # Capture grad mode *before* entering autograd Function
        # (Function.forward runs under torch.no_grad).
        is_grad_enabled = torch.is_grad_enabled()

        checkpoint_activations = (
            self.config.recompute_granularity == 'full' and self.training
        )

        # Execute through single autograd boundary
        h1_out, h2_out = TransformerBlockAutogradFunction.apply(
            h1,
            h2,
            rotary_pos_emb,
            attention_mask,
            self.forward_partitions,
            self.backward_partitions,
            self.forward_tensor_graph,
            self.backward_tensor_graph,
            self.scheduler,
            self.config,
            self.seed_config,
            is_grad_enabled,
            checkpoint_activations,
            *all_params,
        )

        # Concatenate nano-batch outputs
        hidden_states = torch.cat([h1_out, h2_out], dim=1)

        # Final layer norm
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        if self.config.heterogeneous_block_specs:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = get_transformer_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict
