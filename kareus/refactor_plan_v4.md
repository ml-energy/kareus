# Refactoring Plan: Transformer Block Partition Automation

## Current Architecture Summary

The current implementation:
2. **Manually composes** communication operators in `PartitionFuser`, `QKVPartitionFuser`, `AttnOprojPartitionFuser`
3. **Manually defines** partitions with hardcoded first/last layer handling
4. **Each partition** has its own autograd function for gradient computation

## Refactoring Objectives

---

### Objective 1: Detailed TransformerBlock Plan (Primary Entry Point)

**Goal:** TransformerBlock maintains the SAME input/output interface as the current `transformer_block.py`, but internally:
1. Builds a TensorGraph from operators
2. Forms partitions from the graph
3. Executes partitions with proper communication overlap

**Current Interface (to be preserved):**

```python
# kareus/megatron/core/transformer/transformer_block.py

class TransformerBlock(MegatronModule):
    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],  # [s, b, h] where b >= 2 for nano-batch
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
    ) -> Tensor:  # Returns [s, b, h]
        """Same signature as current implementation"""
        ...
```

**New Internal Architecture:**

```python
# kareus/megatron/core/transformer/transformer_block.py

from contextlib import nullcontext
from typing import Optional, Union, List

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules, _get_block_submodules,
)
from megatron.core.parallel_state import (
    get_context_parallel_group, get_context_parallel_world_size,
    get_context_parallel_rank, get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
)
from megatron.core.num_microbatches_calculator import get_micro_batch_size
from megatron.core.extensions.transformer_engine import get_cpu_offload_context, te_checkpoint

from kareus.megatron.core.transformer.transformer_layer import TransformerLayer
from kareus.transformer_engine.pytorch.ops import AllGatherKV, ReduceScatterKV, AllReduce
from kareus.megatron.core.partitions.tensor_graph import (
    TensorGraphBuilder, TensorGraph, CommunicationType,
)
from kareus.megatron.core.partitions.partition_builder import PartitionBuilder
from kareus.megatron.core.partitions.forward_partition import ForwardPartition
from kareus.megatron.core.partitions.backward_partition import BackwardPartition
from kareus.megatron.core.partitions.autograd_function import TransformerBlockAutogradFunction


class TransformerBlock(MegatronModule):
    """Transformer block with automatic partition execution.

    EXTERNAL INTERFACE: Same as original Megatron TransformerBlock.
    INTERNAL EXECUTION: Build graph → Form partitions → Execute with overlap.

    Initialization Flow:
        1. __init__():
           a. Build layers
           b. _build_partitions(): Build TensorGraph, form partitions
              (CommunicationOps are created with operator=None)
        2. set_tensor_parallel_group(): Create AllReduce ops, assign to partitions
        3. set_context_parallel_group(): Create AllGather/ReduceScatter ops, assign to partitions

    Forward Flow:
        1. Split hidden_states into nano-batches (h1, h2)
        2. TransformerBlockAutogradFunction.apply() — executes all partitions
        3. Concatenate outputs, apply final layernorm, return
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

        # Required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        # CPU offloading context (same as upstream Megatron)
        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    self.config.cpu_offloading,
                    self.config.cpu_offloading_num_layers,
                    self.config.num_layers,
                    self.config.cpu_offloading_activations,
                    self.config.cpu_offloading_weights,
                )
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert self.config.cpu_offloading is False, \
                "CPU Offloading is enabled when TE is not present"
            self.offload_context = nullcontext()
            self.group_prefetch_offload_commit_async = None
            self.config._cpu_offloading_context = None

        # Build transformer layers (identical to current _build_layers)
        self._build_layers()

        # Scheduler (set externally by training loop via config)
        self.scheduler = self.config.kareus_scheduler

        # -----------------------------------------------------------------
        # Build tensor graphs and partitions immediately.
        # CommunicationOps are created with operator=None at this point;
        # the physical comm operators are assigned later by
        # set_tensor_parallel_group() / set_context_parallel_group().
        # -----------------------------------------------------------------
        self._build_partitions()

    # =================================================================
    # Layer construction (unchanged from current implementation)
    # =================================================================

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(self.config)
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            fp8_init_context = get_fp8_context(layer_config, global_layer_number - 1, is_init=True)

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
            layers.append(build_layer(layer_spec, i + 1))
        self.layers = torch.nn.ModuleList(layers)

        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None

    # =================================================================
    # Communication initialization
    # Partitions are already built in __init__ with operator=None.
    # These methods create the physical comm operators and assign them
    # to the existing CommunicationOps in all partitions.
    # =================================================================

    def set_tensor_parallel_group(
        self, tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        """Create two shared AllReduce operators and assign to partitions."""
        nano_batch_size = get_micro_batch_size() // 2
        local_seq_length = (
            self.config.max_sequence_length // self.config.context_parallel_size
        )
        hidden_size = self.config.hidden_size

        self.allreduce_comm_ops = []
        for i in range(2):  # two nano-batches
            self.allreduce_comm_ops.append(
                AllReduce(
                    process_group=get_tensor_model_parallel_group(check_initialized=False),
                    async_op=True,
                    backend="msccl",
                    rank=get_tensor_model_parallel_rank(),
                    world_size=get_tensor_model_parallel_world_size(),
                    tensor_size=[local_seq_length, nano_batch_size, hidden_size],
                    device=torch.cuda.current_device(),
                    dtype=torch.bfloat16,
                    batch_idx=i,
                )
            )

        # Assign AllReduce operators to matching CommunicationOps in partitions
        self._assign_comm_operators(CommunicationType.ALL_REDUCE, self.allreduce_comm_ops)

    def set_context_parallel_group(self, cp_group, cp_global_ranks, cp_stream) -> None:
        """Create two shared AllGather/ReduceScatter operators and assign to partitions."""
        nano_batch_size = get_micro_batch_size() // 2
        local_query_groups = (
            self.config.num_query_groups // self.config.tensor_model_parallel_size
        )
        kv_tensor_size = [
            self.config.max_sequence_length,
            nano_batch_size,
            local_query_groups,
            self.config.kv_channels,
        ]

        self.allgather_comm_ops = []
        self.reducescatter_comm_ops = []
        for i in range(2):
            self.allgather_comm_ops.append(
                AllGatherKV(
                    process_group=get_context_parallel_group(check_initialized=False),
                    async_op=True,
                    backend="msccl",
                    rank=get_context_parallel_rank(),
                    world_size=get_context_parallel_world_size(),
                    tensor_size=kv_tensor_size,
                    device=torch.cuda.current_device(),
                    dtype=torch.bfloat16,
                    batch_idx=i,
                )
            )
            self.reducescatter_comm_ops.append(
                ReduceScatterKV(
                    process_group=get_context_parallel_group(check_initialized=False),
                    async_op=True,
                    backend="msccl",
                    rank=get_context_parallel_rank(),
                    world_size=get_context_parallel_world_size(),
                    tensor_size=kv_tensor_size,
                    device=torch.cuda.current_device(),
                    dtype=torch.bfloat16,
                    batch_idx=i,
                )
            )

        # Assign CP operators to matching CommunicationOps in partitions
        self._assign_comm_operators(CommunicationType.ALL_GATHER_KV, self.allgather_comm_ops)
        self._assign_comm_operators(CommunicationType.REDUCE_SCATTER_KV, self.reducescatter_comm_ops)

    # =================================================================
    # Partition building  (NEW — replaces per-layer build_fusers)
    # =================================================================

    def _build_partitions(self):
        """Build TensorGraph and form partitions.

        Called from __init__() after layers are built.

        CommunicationOps in the resulting partitions have operator=None
        at this point. The physical comm operators (AllReduce, AllGatherKV,
        ReduceScatterKV) are assigned later by set_tensor_parallel_group()
        and set_context_parallel_group() via _assign_comm_operators().

        Forward and backward passes have DIFFERENT tensor graphs because:
        1. Backward ops run in reverse order
        2. Backward port semantics are reversed (grad of output → grad of input)
        3. Communication patterns differ (e.g., RowParallel AllReduce in forward,
           ColumnParallel AllReduce in backward)
        """
        # Step 1: Collect all operators from all layers and assign op_id.
        # op_id is a stable identifier on PartitionableOperator, shared by
        # both forward and backward ComputeOps that reference the same operator.
        # Used as key in NanoBatchContext.op_contexts.
        all_ops = []
        for layer in self.layers:
            all_ops.extend(layer.get_all_operators())
        for idx, op in enumerate(all_ops):
            op.op_id = idx

        # Step 2a: Build FORWARD TensorGraph
        forward_graph_builder = TensorGraphBuilder()
        forward_graph_builder.add_initial_channels({
            "main": "t_input_0",
            "rotary_pos_emb": "t_rotary_pos_emb",
            # "bias" / "residual" start unset for the first layer.
            # TensorGraphBuilder returns ext_{name} for missing channels;
            # the first BDA treats None (ext_bias, ext_residual) gracefully.
            # Subsequent layers' channels are written by operators:
            #   - "bias"     ← ColumnParallel/RowParallel ops (return_bias=True)
            #   - "residual" ← ResidualForkOp (before each LayerNorm)
        })
        for op in all_ops:
            for fwd_spec in op.get_forward_ops():
                forward_graph_builder.add_op(fwd_spec)
        forward_tensor_graph = forward_graph_builder.build()

        # Step 2b: Build BACKWARD TensorGraph
        backward_graph_builder = TensorGraphBuilder()
        backward_graph_builder.add_initial_channels({
            "grad_main": "t_grad_output_0",
        })
        for op in reversed(all_ops):
            for bwd_spec in op.get_backward_ops():
                backward_graph_builder.add_op(bwd_spec)
        backward_tensor_graph = backward_graph_builder.build()

        # Store tensor graphs — TransformerBlockAutogradFunction uses them
        # to look up the final tensor_id via get_output_channel("main")
        # and get_output_channel("grad_main").
        self.forward_tensor_graph = forward_tensor_graph
        self.backward_tensor_graph = backward_tensor_graph

        # Step 3: Form partitions (CommunicationOps have operator=None)
        builder = PartitionBuilder(
            forward_tensor_graph=forward_tensor_graph,
            backward_tensor_graph=backward_tensor_graph,
            config=self.config,
        )
        self.forward_partitions = builder.build_forward_partitions()
        self.backward_partitions = builder.build_backward_partitions()

        # Step 4: Assign partition_key for scheduler integration.
        #
        # Each partition_key maps to a field on ScheduleItem (TP-only) or
        # ScheduleItemCP (TP+CP). load_schedule() uses getattr(schedule,
        # partition_key) to fetch the CommConfig for that partition.
        #
        # _form_partitions produces 2 partitions per segment (NB1, NB2),
        # so each key appears twice per layer in the cycle.
        context_parallel = self.config.context_parallel_size > 1

        if not context_parallel:
            # TP-only: 2 segments/layer (attn, mlp) × 2 NB = 4 partitions/layer
            #   ScheduleItem fields: fwd_attn, fwd_mlp, bwd_attn, bwd_mlp
            fwd_keys_per_layer = ["fwd_attn", "fwd_attn", "fwd_mlp", "fwd_mlp"]
            bwd_keys_per_layer = ["bwd_mlp", "bwd_mlp", "bwd_attn", "bwd_attn"]
        else:
            # TP+CP: more segments due to separate CP comm ops.
            #   ScheduleItemCP fields cycle per layer.
            #
            # Forward segments (in execution order):
            #   QKV→AllReduce(TP), QKV→AllGather(CP),
            #   Attn+OProj→AllGather(CP), Attn+OProj→AllReduce(TP),
            #   MLP→AllReduce(TP)
            fwd_keys_per_layer = [
                "fwd_qkv_ar", "fwd_qkv_ar",
                "fwd_qkv_ag", "fwd_qkv_ag",
                "fwd_ao_ag",  "fwd_ao_ag",
                "fwd_ao_ar",  "fwd_ao_ar",
                "fwd_mlp",    "fwd_mlp",
            ]
            # Backward segments (reverse of forward):
            #   MLP→AllReduce(TP),
            #   OProj→AllReduce(TP), OProj→AllGather(CP),
            #   Attn→AllGather(CP), Attn→ReduceScatter(CP),
            #   QKV→ReduceScatter(CP), QKV→AllReduce(TP)
            bwd_keys_per_layer = [
                "bwd_mlp",    "bwd_mlp",
                "bwd_o_ar",   "bwd_o_ar",
                "bwd_o_ag",   "bwd_o_ag",
                "bwd_a_ag",   "bwd_a_ag",
                "bwd_a_rs",   "bwd_a_rs",
                "bwd_qkv_rs", "bwd_qkv_rs",
                "bwd_qkv_ar", "bwd_qkv_ar",
            ]

        num_layers = len(self.layers)
        fwd_keys = fwd_keys_per_layer * num_layers
        bwd_keys = bwd_keys_per_layer * num_layers

        assert len(self.forward_partitions) == len(fwd_keys), (
            f"Forward partition count {len(self.forward_partitions)} != "
            f"expected {len(fwd_keys)} ({len(fwd_keys_per_layer)}/layer × {num_layers} layers)"
        )
        assert len(self.backward_partitions) == len(bwd_keys), (
            f"Backward partition count {len(self.backward_partitions)} != "
            f"expected {len(bwd_keys)} ({len(bwd_keys_per_layer)}/layer × {num_layers} layers)"
        )

        for partition, key in zip(self.forward_partitions, fwd_keys):
            partition.partition_key = key
        for partition, key in zip(self.backward_partitions, bwd_keys):
            partition.partition_key = key

    def _assign_comm_operators(
        self,
        comm_type: CommunicationType,
        comm_ops: List,  # [op_nb0, op_nb1]
    ):
        """Assign physical comm operators to all CommunicationOps of a given type.

        Iterates over all forward and backward partitions. For each partition
        whose comm_op matches ``comm_type``, assigns the physical operator
        from ``comm_ops`` indexed by nano_batch_idx.

        Args:
            comm_type: The CommunicationType to match (ALL_REDUCE, ALL_GATHER_KV, etc.)
            comm_ops: List of two physical operator instances [nb0, nb1].
        """
        for partition in self.forward_partitions + self.backward_partitions:
            if partition.comm_op is not None and partition.comm_op.comm_type == comm_type:
                partition.comm_op.operator = comm_ops[partition.nano_batch_idx]

    # =================================================================
    # Pipeline-parallel helpers (unchanged)
    # =================================================================

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it.
        """
        self.input_tensor = input_tensor

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    # =================================================================
    # Nano-batch splitting (unchanged from current implementation)
    # =================================================================

    def _split_tensors_for_nanobatch(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> tuple:
        """Split input tensors along the batch dimension into two nano-batches."""
        batch_size = hidden_states.size(1)
        if batch_size < 2:
            raise ValueError(
                f"Batch size must be at least 2 for nano-batch splitting, got {batch_size}"
            )
        mid_point = batch_size // 2

        # Currently only hidden_states is split; masks are asserted None.
        assert attention_mask is None
        assert context is None
        assert context_mask is None
        assert attention_bias is None
        assert sequence_len_offset is None

        h1 = hidden_states[:, :mid_point, ...]
        h2 = hidden_states[:, mid_point:, ...]
        return (
            (h1, None, None, None, None, None),
            (h2, None, None, None, None, None),
        )

    # =================================================================
    # Forward (public interface — unchanged signature)
    # =================================================================

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
    ) -> Tensor:
        """Forward pass — SAME INTERFACE as original TransformerBlock.

        Internal execution:
        1. Split into nano-batches
        2. Execute all partitions via TransformerBlockAutogradFunction
        3. Concatenate outputs, apply final layernorm, return
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True,
        )

        # Split into two nano-batches along the batch dimension
        (h1, *_), (h2, *_) = self._split_tensors_for_nanobatch(
            hidden_states, attention_mask, context, context_mask,
            attention_bias, sequence_len_offset,
        )

        # Flatten all layer parameters for autograd tracking.
        # TransformerBlockAutogradFunction receives them as *params so that
        # PyTorch's autograd engine knows to compute their gradients.
        all_params = list(self._get_all_params())

        # Execute the entire block through a single autograd boundary
        h1_out, h2_out = TransformerBlockAutogradFunction.apply(
            h1, h2,
            rotary_pos_emb,
            attention_mask,
            self.forward_partitions,
            self.backward_partitions,
            self.forward_tensor_graph,
            self.backward_tensor_graph,
            self.scheduler,
            self.config,
            *all_params,
        )

        # Concatenate nano-batch outputs back into full batch
        hidden_states = torch.cat([h1_out, h2_out], dim=1)

        # Final layer norm (applied to the full batch, outside partitions)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True,
            )

        return hidden_states

    # =================================================================
    # Parameter collection
    # =================================================================

    def _get_all_params(self) -> List[torch.nn.Parameter]:
        """Get all parameters from all layers for autograd tracking.

        The order MUST be deterministic and match the order used by
        _combine_param_grads in the autograd function backward.
        Using layer.parameters() (which iterates sub-modules in
        registration order) guarantees this.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    # =================================================================
    # State dict (unchanged from current implementation)
    # =================================================================

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None,
    ) -> ShardedStateDict:
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
        for i, layer in enumerate(self.layers):
            global_layer_offset = layer.layer_number - 1
            state_dict_prefix = f'{layer_prefix}{i}.'
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [(0, global_layer_offset, num_layers)]
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata,
            )
            replace_prefix_for_sharding(
                layer_sharded_state_dict, state_dict_prefix, sharded_prefix,
            )
            sharded_state_dict.update(layer_sharded_state_dict)

        for name, module in self.named_children():
            if module is not self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata,
                    )
                )
        return sharded_state_dict
```

**Execution Flow Diagram:**

```
TransformerBlock.forward()
    │
    ├── 1. Split hidden_states → (h1, h2)
    │
    ├── 2. TransformerBlockAutogradFunction.apply()
    │       │
    │       ├── Create NanoBatchContexts:
    │       │       ctx_nb1 = NanoBatchContext(batch_idx=0)
    │       │       ctx_nb2 = NanoBatchContext(batch_idx=1)
    │       │       ctx_nb1.tensor_store.set("t_input_0", h1)
    │       │       ctx_nb2.tensor_store.set("t_input_0", h2)
    │       │       # Seed external inputs (shared across both nano-batches)
    │       │       ctx_nb1.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)
    │       │       ctx_nb2.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)
    │       │
    │       ├── Load schedule: scheduler.current_schedule
    │       │
    │       ├── For each ForwardPartition (interleaved NB1/NB2):
    │       │       │
    │       │       ├── partition.load_schedule(schedule)
    │       │       │
    │       │       └── if partition.nano_batch_idx == 0:
    │       │           │   partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
    │       │           │   # comp_ops read/write ctx_nb1.tensor_store
    │       │           │   # comm_op  read/write ctx_nb2.tensor_store
    │       │           else:
    │       │               partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
    │       │               # comp_ops read/write ctx_nb2.tensor_store
    │       │               # comm_op  read/write ctx_nb1.tensor_store
    │       │
    │       ├── final_tensor_id = forward_tensor_graph.get_output_channel("main")
    │       │   h1_out = ctx_nb1.tensor_store.get(final_tensor_id)
    │       │   h2_out = ctx_nb2.tensor_store.get(final_tensor_id)
    │       │
    │       ├── Save for backward:
    │       │       ctx_nb1.flatten_saved_tensors()
    │       │       ctx_nb2.flatten_saved_tensors()
    │       │       func_ctx.save_for_backward(...)
    │       │
    │       └── Return (h1_out, h2_out)
    │
    └── 3. Concatenate → hidden_states

TransformerBlockAutogradFunction.backward()
    │
    ├── Restore saved tensors:
    │       ctx_nb1.restore_saved_tensors(func_ctx.saved_tensors[...])
    │       ctx_nb2.restore_saved_tensors(func_ctx.saved_tensors[...])
    │
    ├── Seed backward tensor_stores with grad_output:
    │       ctx_nb1.tensor_store.set("t_grad_output_0", grad_h1)
    │       ctx_nb2.tensor_store.set("t_grad_output_0", grad_h2)
    │
    ├── For each BackwardPartition (interleaved NB1/NB2):
    │       │
    │       └── if partition.nano_batch_idx == 0:
    │           │   grad_params = partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
    │           else:
    │               grad_params = partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
    │
    ├── Merge parameter grads from both nanobatches (sum)
    │
    └── Return (grad_h1, grad_h2, *grad_params_flat)
```

---

### Objective 2: Automatic Tensor Graph and Communication Interface

**Files to Create/Modify:**
- `kareus/megatron/core/partitions/tensor_graph.py` (NEW - automatic tensor naming)
- `kareus/megatron/core/partitions/operator.py` (NEW)
- `kareus/megatron/core/extensions/ops/*.py` (MODIFY - add interface)

**Key Design Principles:**

1. **NAMED CHANNEL ROUTING**: Each operator declares named semantic channels (e.g., `"main"`, `"bias"`, `"key"`) for its
   inputs and outputs. The graph builder routes tensors by matching channel names, enabling non-adjacent connections
   (channels persist in the registry until overwritten by a later operator).

2. **AUTOMATIC BACKWARD DERIVATION**: Operators only declare FORWARD channels. Backward channels are auto-derived by
   `ComputeOpSpec(is_backward=True)`: input/output channels are reversed, names are prefixed with `"grad_"`.

3. **EXPLICIT RESIDUAL FORK/ACCUMULATE**: Residual connections are handled by `ResidualForkOp` — an explicit operator
   placed before each LayerNorm. Forward: `x → (x_main, x_residual)`. Backward: `grad_main + grad_residual → grad_input`.

4. **AUTOMATIC TENSOR NAMING**: Tensors are automatically assigned unique IDs (`t_0`, `t_1`, ...) by `TensorGraphBuilder`
   at graph-build time. Channel names are a build-time routing concept; at runtime, everything uses tensor IDs.

#### Channel (`Channel`)

A **channel** is a **named semantic slot** on an operator (e.g., `"main"`, `"bias"`, `"key"`, `"residual"`).
Each channel has a `port_idx` that maps to the positional slot in `fuser_forward`/`fuser_backward` calls.

- **Channel `"main"` (port 0)** is always the primary tensor (hidden_states / grad)
- **Additional channels** carry side-channel tensors (key, value, bias, residual, etc.)

Channels **persist** in the `TensorGraphBuilder`'s registry until overwritten by a later operator,
enabling non-adjacent connections. For example, `"value"` written by `QKVPostProcessOp` persists
through `RotaryEmbeddingOp` (which doesn't write `"value"`) and is read by `DotProductAttentionOp`.

#### Tensor ID (`tensor_id`)

A **tensor ID** is a **unique identifier** assigned by `TensorGraphBuilder` to each actual tensor 
flowing through the graph. It's auto-generated at graph-building time (e.g., `t_0`, `t_1`, `t_2`, ...).

Tensor IDs are used to:
- Route tensors between operators within a partition
- Pass tensors between partitions (cross-nanobatch flow)
- Store/retrieve tensors in `TensorStore`

#### ResidualForkOp

An explicit `ResidualForkOp` is placed **before each LayerNorm** in the operator
sequence (see `TransformerLayer.get_all_operators()`):

```
BDA → ResidualForkOp → LayerNorm → ...
```

- **Forward:** `x → (x_main, x_residual)` — identity + copy. `x_main` feeds LayerNorm,
  `x_residual` persists in the `"residual"` channel for the next BDA.
- **Backward:** `(grad_main, grad_residual) → grad_main + grad_residual` — the mathematically
  correct gradient at a fork point. The accumulation is a real op, not a graph-builder side-effect.

---


**Design:**

```python
# kareus/megatron/core/partitions/tensor_graph.py

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
from abc import ABC, abstractmethod

class CommunicationType(Enum):
    """Types of communication operations"""
    ALL_REDUCE = auto()       # 1 input, 1 output
    ALL_GATHER_KV = auto()    # 2 inputs (k, v), 2 outputs (gathered_k, gathered_v)
    REDUCE_SCATTER_KV = auto() # 2 inputs (grad_k, grad_v), 2 outputs (scattered_grad_k, scattered_grad_v)

@dataclass
class Channel:
    """
    Named connection point on an operator.
    
    Used for BOTH input and output declarations. The channel name identifies
    the semantic meaning of the tensor, while port_idx maps to the positional
    slot in fuser_forward/fuser_backward calls.
    
    Examples:
        Channel(0, "main")      — primary hidden_states tensor
        Channel(1, "bias")      — bias from Linear op
        Channel(1, "key")       — key from QKV decomposition
        Channel(1, "residual")  — residual from ResidualForkOp
        Channel(2, "value")     — value from QKV decomposition
    """
    port_idx: int
    name: str


## ChannelEffect is REMOVED — replaced by ResidualForkOp (see above)


@dataclass
class TensorPort:
    """
    Represents a single input or output port of an operator.
    
    Ports are numbered (0, 1, 2...) and automatically connected by the graph builder.
    Port 0 is always the main hidden_states tensor.
    Additional ports are for extra tensors (key, value, bias, residual, etc.)
    """
    port_idx: int
    tensor_id: Optional[str] = None  # Assigned by graph builder, e.g., "t_0", "t_1"
    
    def __repr__(self):
        return f"Port({self.port_idx}, tensor={self.tensor_id})"


@dataclass
class ComputeOp:
    """
    Represents a computation operation in the graph.
    """
    operator: 'PartitionableOperator'
    input_ports: List[TensorPort] = field(default_factory=list)
    output_ports: List[TensorPort] = field(default_factory=list)
    
    def get_input_tensor_ids(self) -> List[str]:
        """Get tensor IDs for all input ports"""
        return [p.tensor_id for p in self.input_ports]
    
    def get_output_tensor_ids(self) -> List[str]:
        """Get tensor IDs for all output ports"""
        return [p.tensor_id for p in self.output_ports]


@dataclass
class CommunicationOp:
    """
    Represents a communication operation in the graph.
    
    Port count is determined by comm_type:
    - ALL_REDUCE: 1 port (hidden_states)
    - ALL_GATHER_KV: 2 ports (key, value)
    - REDUCE_SCATTER_KV: 2 ports (grad_key, grad_value)
    """
    comm_type: CommunicationType
    operator: Optional[Any] = None  # Shared operator instance, assigned by transformer_block
    input_ports: List[TensorPort] = field(default_factory=list)
    output_ports: List[TensorPort] = field(default_factory=list)
    
    def get_input_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.input_ports]
    
    def get_output_tensor_ids(self) -> List[str]:
        return [p.tensor_id for p in self.output_ports]


class PartitionableOperator(ABC):
    """
    Base interface for operators that participate in partitioning.
    
    KEY DESIGN: Operators declare NAMED CHANNELS for forward I/O only.
    Backward channels are auto-derived by ComputeOpSpec(is_backward=True):
      - Input channels  = [grad_{ch} for ch in forward output_channels]
      - Output channels = [grad_{ch} for ch in forward input_channels]
    
    The TensorGraphBuilder routes tensors by channel NAME (not port index).
    Channels persist in the registry until overwritten, enabling non-adjacent
    connections (e.g., "value" from QKVPostProcess skipping RotaryEmbed to
    reach DotProductAttention).
    
    op_id: Unique identifier assigned by TransformerBlock._build_partitions()
    during Step 1 (operator collection). Used as key in NanoBatchContext.op_contexts
    for saving/restoring OperationContext across forward and backward passes.
    Since both forward and backward ComputeOps reference the same operator
    instance, they share the same op_id automatically.
    """
    
    op_id: int = -1  # Assigned by _build_partitions Step 1
    
    @abstractmethod
    def get_forward_ops(self) -> List[Union['ComputeOpSpec', 'CommunicationOpSpec']]:
        """
        Return operation specifications for forward pass.
        
        Returns list of ComputeOpSpec or CommunicationOpSpec.
        The graph builder will instantiate actual ComputeOp/CommunicationOp
        with proper channel connections.
        """
        pass
    
    @abstractmethod
    def get_backward_ops(self) -> List[Union['ComputeOpSpec', 'CommunicationOpSpec']]:
        """
        Return operation specifications for backward pass.
        """
        pass
    
    def get_input_channels(self) -> List['Channel']:
        """
        Declare which named channels each input port reads from (forward).
        
        Channel(0, "main") is always present (primary hidden_states).
        Override to declare additional channels.
        
        Default: [Channel(0, "main")]
        """
        return [Channel(0, "main")]
    
    def get_output_channels(self) -> List['Channel']:
        """
        Declare which named channels each output port writes to (forward).
        
        Channel(0, "main") is always present (primary hidden_states).
        Override to declare additional channels.
        
        Default: [Channel(0, "main")]
        """
        return [Channel(0, "main")]


@dataclass
class ComputeOpSpec:
    """
    Specification for creating a ComputeOp.
    
    IMPORTANT: For backward ops, set is_backward=True.
    This automatically reverses the channel semantics:
    - Input channels  = grad of forward output channels (prefixed with "grad_")
    - Output channels = grad of forward input channels (prefixed with "grad_")
    
    This mirrors the mathematical structure of backpropagation:
    - Backward "input"  = gradient of forward output
    - Backward "output" = gradient of forward input
    """
    operator: PartitionableOperator
    is_backward: bool = False
    
    def get_input_channels(self) -> List[Channel]:
        if self.is_backward:
            # Backward input = grad of forward output
            return [Channel(i, f"grad_{ch.name}")
                    for i, ch in enumerate(self.operator.get_output_channels())]
        return self.operator.get_input_channels()
    
    def get_output_channels(self) -> List[Channel]:
        if self.is_backward:
            # Backward output = grad of forward input
            return [Channel(i, f"grad_{ch.name}")
                    for i, ch in enumerate(self.operator.get_input_channels())]
        return self.operator.get_output_channels()


@dataclass
class CommunicationOpSpec:
    """
    Specification for creating a CommunicationOp.
    
    Communication ops declare channels directly (no forward/backward reversal
    needed since forward and backward graphs have separate comm ops).
    """
    comm_type: CommunicationType
    channels: List[Channel]  # e.g., [Channel(0, "main")] or [Channel(0, "key"), Channel(1, "value")]
    
    def get_input_channels(self) -> List[Channel]:
        return self.channels
    
    def get_output_channels(self) -> List[Channel]:
        return self.channels  # Comm ops read and write the same channels


class TensorGraphBuilder:
    """
    Builds a tensor dependency graph using NAMED CHANNEL routing.
    
    Key differences from a simple linear pipeline:
    1. Channel registry persists across operators (NOT cleared after each op)
    2. Operators connect via semantic channel names, not port indices
    3. ResidualForkOp handles fork/accumulate as explicit ops (no side-effects)
    4. Non-adjacent connections work naturally (channels persist until overwritten)
    
    Example channel flow for attention partition:
        BDA:            reads main, bias, residual → writes main
        ResidualFork:   reads main                 → writes main, residual
        LN:             reads main                 → writes main
        QKVPost:        reads main                 → writes main(=query), key, value
        Rotary:         reads main, key, rotary_pos_emb → writes main, key
        CoreAttn:       reads main, key, value     → writes main
        (value persists from QKVPost because Rotary doesn't overwrite it)
        (residual persists from ResidualFork because LN/QKV/etc. don't overwrite it)
    """
    
    def __init__(self):
        self._next_tensor_id = 0
        self._ops: List[Union[ComputeOp, CommunicationOp]] = []
        
        # Named channel registry: channel_name → tensor_id
        # Persists across operators — channels are NOT cleared after each op.
        self._channel_registry: Dict[str, str] = {}
        
    def _new_tensor_id(self) -> str:
        tid = f"t_{self._next_tensor_id}"
        self._next_tensor_id += 1
        return tid
    
    def add_initial_channels(self, channels: Dict[str, str]):
        """
        Seed the channel registry with initial/external tensors.
        
        Called before adding any ops. Sets up the initial state:
          - "main" → input hidden_states tensor_id
          - "bias" → bias from previous partition (if any)
          - "residual" → residual from previous partition (if any)
          - "rotary_pos_emb" → external rotary embedding tensor_id
        
        For backward graphs, use grad_ prefixed names:
          - "grad_main" → grad_output tensor_id
          - "grad_bias" → grad_bias from next partition's backward
          - "grad_residual" → grad_residual from next partition's backward
        
        Example:
            builder.add_initial_channels({
                "main": "t_input_0",
                "rotary_pos_emb": "t_rope",
            })
        """
        for channel_name, tensor_id in channels.items():
            self._channel_registry[channel_name] = tensor_id
    
    def add_op(self, spec: Union[ComputeOpSpec, CommunicationOpSpec]) -> Union[ComputeOp, CommunicationOp]:
        """
        Add an operator to the graph using named channel routing.
        
        Steps:
        1. Wire input ports from channel registry by channel NAME
        2. Create output ports with new tensor IDs, update channel registry
        
        NOTE: Channel registry is NOT cleared — channels persist for later ops.
        """
        # --- Step 1: Wire input ports from channel registry ---
        input_ports = []
        for ch in spec.get_input_channels():
            tensor_id = self._channel_registry.get(ch.name)
            if tensor_id is None:
                tensor_id = f"ext_{ch.name}"  # External/missing tensor
            input_ports.append(TensorPort(port_idx=ch.port_idx, tensor_id=tensor_id))
        
        # --- Step 2: Create output ports, update channel registry ---
        # NOTE: We do NOT clear the registry. Channels persist until overwritten.
        output_ports = []
        for ch in spec.get_output_channels():
            tensor_id = self._new_tensor_id()
            output_ports.append(TensorPort(port_idx=ch.port_idx, tensor_id=tensor_id))
            self._channel_registry[ch.name] = tensor_id  # Overwrite channel
        
        # --- Instantiate op ---
        if isinstance(spec, ComputeOpSpec):
            op = ComputeOp(
                operator=spec.operator,
                input_ports=input_ports,
                output_ports=output_ports,
            )
        else:  # CommunicationOpSpec
            op = CommunicationOp(
                comm_type=spec.comm_type,
                input_ports=input_ports,
                output_ports=output_ports,
            )
        
        self._ops.append(op)
        return op
    
    def build(self) -> 'TensorGraph':
        """Build and return the complete tensor graph"""
        return TensorGraph(
            ops=self._ops,
            channel_registry=self._channel_registry,
        )
    
    def get_channel_registry(self) -> Dict[str, str]:
        """Get current channel→tensor_id mapping (useful for partition boundaries)."""
        return self._channel_registry


@dataclass
class TensorGraph:
    """
    Complete tensor dependency graph.
    
    Contains all ops (compute and comm) with their channel-routed port connections.
    Used by partition builder and executor to route tensors correctly.
    """
    ops: List[Union[ComputeOp, CommunicationOp]]
    channel_registry: Dict[str, str]  # Final channel→tensor_id mapping
    
    def get_compute_ops(self) -> List[ComputeOp]:
        return [op for op in self.ops if isinstance(op, ComputeOp)]
    
    def get_comm_ops(self) -> List[CommunicationOp]:
        return [op for op in self.ops if isinstance(op, CommunicationOp)]
    
    def get_output_channel(self, channel_name: str) -> Optional[str]:
        """Get the final tensor_id for a named channel."""
        return self.channel_registry.get(channel_name)
```


**Example Implementation for Operators:**

```python
# kareus/megatron/core/extensions/ops/te_linear.py (modify)

class TEColumnParallelLinearOp(TELinearOp, PartitionableOperator):
    """
    Column parallel linear with communication interface.
    
    Used by: QKV projection (linear_qkv), MLP first layer (linear_fc1).
    
    Forward:  compute only (input is replicated, output is partitioned — no comm needed)
    Backward: compute → AllReduce (gradient of replicated input is partial sum → needs AllReduce)
    
    Channels: reads "main", writes "main" and optionally "bias" (if return_bias=True).
    
    The backward AllReduce is CRITICAL for correctness in tensor parallelism:
      Forward:  y_partitioned = x_replicated @ W_partitioned^T  (no comm)
      Backward: grad_x = AllReduce(grad_y @ W_partitioned)      (partial sum → AllReduce)
    
    Without this backward AllReduce, backward partitions would have NO communication
    boundaries, preventing nano-batch overlap in the backward pass.
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_output_channels(self) -> List[Channel]:
        if self.return_bias:
            return [Channel(0, "main"), Channel(1, "bias")]
        return [Channel(0, "main")]
    
    def get_forward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        # No communication in forward — output is partitioned, consumed locally
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        # Backward: compute produces partial-sum gradient → AllReduce to get full gradient
        # This AllReduce is the partition boundary in backward (mirrors forward's
        # RowParallel AllReduce partition boundary).
        return [
            ComputeOpSpec(operator=self, is_backward=True),
            CommunicationOpSpec(
                comm_type=CommunicationType.ALL_REDUCE,
                channels=[Channel(0, "grad_main")],
            ),
        ]


class TERowParallelLinearOp(TELinearOp, PartitionableOperator):
    """
    Row parallel linear with communication interface.
    
    Used by: output projection (linear_proj), MLP second layer (linear_fc2).
    
    Forward: compute → AllReduce (output is partial sum → needs AllReduce)
    Backward: compute only (gradient is partitioned, feeds into ColumnParallel backward)
    
    Channels: reads "main", writes "main" and optionally "bias" (if return_bias=True).
    The "bias" channel persists in the registry until consumed by the next
    BiasDropoutAddOp (which may be many operators later, possibly in the next layer).
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_output_channels(self) -> List[Channel]:
        if self.return_bias:
            return [Channel(0, "main"), Channel(1, "bias")]
        return [Channel(0, "main")]
    
    def get_forward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        return [
            ComputeOpSpec(operator=self),  # is_backward=False (default)
            CommunicationOpSpec(
                comm_type=CommunicationType.ALL_REDUCE,
                channels=[Channel(0, "main")],
            ),
        ]
    
    def get_backward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        # No communication in backward — gradient is partitioned and feeds
        # directly into ColumnParallel backward (which will AllReduce)
        # is_backward=True auto-derives backward channels:
        #   input:  [Channel(0, "grad_main")]  (from forward output "main")
        #   output: [Channel(0, "grad_main")]  (from forward input "main")
        return [ComputeOpSpec(operator=self, is_backward=True)]


# --- Additional operator examples demonstrating channel patterns ---


class ResidualForkOp(PartitionableOperator):
    """
    Residual fork: duplicates input tensor for residual connection.
    Placed BEFORE each LayerNorm in the operator sequence.
    
    Forward:  x → (x_main, x_residual)
        - x_main goes to LayerNorm (main path)
        - x_residual persists in "residual" channel for next BDA
    
    Backward: (grad_main, grad_residual) → grad_main + grad_residual
        - The mathematically correct gradient at a fork point.
        - Replaces the old ChannelEffect("accumulate") approach.
    
    Auto-derived backward:
      input:  [Channel(0, "grad_main"), Channel(1, "grad_residual")]
      output: [Channel(0, "grad_main")]
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_output_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),       # goes to LayerNorm
            Channel(1, "residual"),   # saved for next BDA
        ]
    
    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class LayerNormOp(PartitionableOperator):
    """
    LayerNorm — simple main→main operator.
    
    Residual forking is handled by ResidualForkOp (placed before this op),
    NOT by pre-effects on LayerNorm.
    
    Forward:  reads "main", writes "main"
    Backward: reads "grad_main", writes "grad_main"
    """
    
    # Channels: default main→main, no need to override
    
    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class BiasDropoutAddOp(PartitionableOperator):
    """
    BiasDropoutAdd: output = residual + dropout(input + bias)
    
    Consumes bias and residual from the channel registry.
    These may come from non-adjacent operators (e.g., bias from a Bias op
    several operators ago, residual from ResidualForkOp before the
    previous LayerNorm).
    
    Auto-derived backward:
      input:  [Channel(0, "grad_main")]
      output: [Channel(0, "grad_main"), Channel(1, "grad_bias"), Channel(2, "grad_residual")]
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),
            Channel(1, "bias"),       # from Bias op (may be many ops earlier)
            Channel(2, "residual"),   # from ResidualForkOp (before previous LN)
        ]
    
    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]


class RotaryEmbeddingOp(PartitionableOperator):
    """
    Rotary positional embedding: applies RoPE to query and key.
    
    Reads "rotary_pos_emb" from the channel registry — this is an EXTERNAL input
    seeded via add_initial_channels("rotary_pos_emb", "t_rotary_pos_emb") and
    set in each NanoBatchContext's tensor_store before execution.
    
    The "rotary_pos_emb" channel persists from the initial seeding because no
    operator overwrites it. It is read by every RotaryEmbeddingOp in every layer.
    
    Forward:  reads main (query), key, rotary_pos_emb → writes main (query), key
    Backward: reads grad_main, grad_key → writes grad_main, grad_key
              (rotary_pos_emb is not differentiated — no grad_rotary_pos_emb output)
    
    NOTE: rotary_pos_emb is treated as a non-differentiable external input.
    The backward auto-derivation would produce Channel("grad_rotary_pos_emb"),
    but since no downstream op reads it, the gradient is simply unused.
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),            # query
            Channel(1, "key"),             # from QKVPostProcess
            Channel(2, "rotary_pos_emb"),  # external input (persists from initial seeding)
        ]
    
    def get_output_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),   # rotated query
            Channel(1, "key"),    # rotated key
        ]
    
    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]
        # Auto-derived backward:
        #   input:  [Channel(0, "grad_main"), Channel(1, "grad_key")]
        #   output: [Channel(0, "grad_main"), Channel(1, "grad_key"), Channel(2, "grad_rotary_pos_emb")]
        #   grad_rotary_pos_emb is produced but unused (no op reads it)


class QKVPostProcessOp(PartitionableOperator):
    """
    QKV post-processing: splits mixed_qkv into query, key, value.
    
    Forward:  1 input  → 3 outputs (main=query, key, value)
    Backward: 3 inputs (grad_main, grad_key, grad_value) → 1 output (auto-derived)
    """
    
    def get_input_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_output_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),   # query
            Channel(1, "key"),
            Channel(2, "value"),
        ]
    
    def get_forward_ops(self):
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self):
        return [ComputeOpSpec(operator=self, is_backward=True)]
        # Auto-derived backward:
        #   input:  [Channel(0, "grad_main"), Channel(1, "grad_key"), Channel(2, "grad_value")]
        #   output: [Channel(0, "grad_main")]


class DotProductAttentionOp(PartitionableOperator):
    """
    Dot-product attention: reads query (main), key, value.
    
    key comes from RotaryEmbed, value comes from QKVPostProcess.
    The "value" channel PERSISTS through RotaryEmbed (which doesn't write it),
    demonstrating non-adjacent channel connections.
    
    Auto-derived backward:
      input:  [Channel(0, "grad_main")]
      output: [Channel(0, "grad_main"), Channel(1, "grad_key"), Channel(2, "grad_value")]
    
    Context Parallelism (CP > 1):
      When CP is enabled, key and value must be gathered across the CP group
      before attention, and gradients must be reduce-scattered after backward.
      
      Forward:  AllGatherKV(key, value) → compute
      Backward: AllGatherKV(grad_key, grad_value) → compute → ReduceScatterKV(grad_key, grad_value)
      
      The AllGatherKV BEFORE compute gathers full-sequence KV from all CP ranks.
      The backward AllGatherKV gathers grad_key/grad_value for local backward compute.
      The backward ReduceScatterKV distributes the computed KV gradients back to CP ranks.
    """
    
    def __init__(self, ..., use_cp: bool = False):
        ...
        self.use_cp = use_cp
    
    def get_input_channels(self) -> List[Channel]:
        return [
            Channel(0, "main"),   # query (from RotaryEmbed)
            Channel(1, "key"),    # from RotaryEmbed (modified key)
            Channel(2, "value"),  # from QKVPostProcess (persisted through Rotary!)
        ]
    
    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main")]
    
    def get_forward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        if self.use_cp:
            # CP mode: gather full-sequence KV before attention
            return [
                CommunicationOpSpec(
                    comm_type=CommunicationType.ALL_GATHER_KV,
                    channels=[Channel(0, "key"), Channel(1, "value")],
                ),
                ComputeOpSpec(operator=self),
            ]
        return [ComputeOpSpec(operator=self)]
    
    def get_backward_ops(self) -> List[Union[ComputeOpSpec, CommunicationOpSpec]]:
        if self.use_cp:
            # CP backward:
            #   1. AllGatherKV: gather grad_key/grad_value from all CP ranks
            #      (needed because backward compute requires full-sequence KV context)
            #   2. Compute: local backward producing grad_query, grad_key, grad_value
            #   3. ReduceScatterKV: distribute KV gradients back to CP ranks
            return [
                CommunicationOpSpec(
                    comm_type=CommunicationType.ALL_GATHER_KV,
                    channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
                ),
                ComputeOpSpec(operator=self, is_backward=True),
                CommunicationOpSpec(
                    comm_type=CommunicationType.REDUCE_SCATTER_KV,
                    channels=[Channel(0, "grad_key"), Channel(1, "grad_value")],
                ),
            ]
        return [ComputeOpSpec(operator=self, is_backward=True)]
```

---


### Objective 3: Automatic Partition Formation

**Key Insight: Interleaved Nanobatch Partition Formation**
**Partitions are formed by interleaving computations with communications from the other nanobatch:**

**Files to Create:**
- `kareus/megatron/core/partitions/partition_builder.py` (NEW)

**Design:**

```python
# kareus/megatron/core/partitions/partition_builder.py

class PartitionBuilder:
    """
    Automatically forms partitions from already-built tensor graphs.
    
    Takes SEPARATE forward and backward TensorGraphs (built by
    TransformerBlock._build_partitions) and forms ForwardPartition
    and BackwardPartition instances.
    
    The tensor graphs already contain fully-wired ComputeOp and
    CommunicationOp instances with correct port connections:
    - Forward graph: ports follow forward semantics
    - Backward graph: ports are reversed (grad of output → grad of input)
      because ComputeOpSpec(is_backward=True) was used when building
    """
    
    def __init__(
        self,
        forward_tensor_graph: TensorGraph,
        backward_tensor_graph: TensorGraph,
        config: TransformerConfig,
    ):
        self.forward_tensor_graph = forward_tensor_graph
        self.backward_tensor_graph = backward_tensor_graph
        self.config = config
    
    def build_forward_partitions(self) -> List[ForwardPartition]:
        """
        Build ForwardPartition instances from forward tensor graph.
        
        Uses the already-built ops from forward_tensor_graph (which have
        correct forward port connections).
        """
        forward_ops = self.forward_tensor_graph.ops  # Already built ComputeOp/CommunicationOp
        return self._form_partitions(forward_ops, ForwardPartition)
    
    def build_backward_partitions(self) -> List[BackwardPartition]:
        """
        Build BackwardPartition instances from backward tensor graph.
        
        Uses the already-built ops from backward_tensor_graph (which have
        reversed port connections via is_backward=True).
        """
        backward_ops = self.backward_tensor_graph.ops  # Already built ComputeOp/CommunicationOp
        return self._form_partitions(backward_ops, BackwardPartition)
    
    def _form_partitions(
        self,
        ops: List[Union[ComputeOp, CommunicationOp]],
        partition_class: type,  # ForwardPartition or BackwardPartition
    ) -> List[Union[ForwardPartition, BackwardPartition]]:
        """
        Form partitions from a sequence of ops.
        
        Given ops: [A, AR, B, AR] where A, B are ComputeOps and AR is CommunicationOp
        
        With two nanobatches, the interleaved sequence is:
        NB1: A1 AR1 B1 AR1
        NB2: A2 AR2 B2 AR2
        
        Partitions formed:
        P0: (comp_ops=[A], comm=None)       - NB1 seg0, no comm to wait (first)
        P1: (comp_ops=[A], comm=AR_seg0)    - NB2 seg0, wait AR from NB1's seg0
        P2: (comp_ops=[B], comm=AR_seg0)    - NB1 seg1, wait AR from NB2's seg0
        P3: (comp_ops=[B], comm=AR_seg1)    - NB2 seg1, wait AR from NB1's seg1
        
        Key insight:
        - NB1 partitions wait for NB2's comm from PREVIOUS segment
        - NB2 partitions wait for NB1's comm from CURRENT segment (just started)
        """
        partitions = []
        
        # Group ops into segments separated by communications
        segments = self._split_by_communications(ops)
        # segments = [(comp_ops, comm_after), (comp_ops, comm_after), ...]
        
        partition_id = 0
        prev_comm = None  # comm_after from previous segment (used by NB1)
        
        for seg_idx, (comp_ops, comm_after) in enumerate(segments):
            # Nanobatch 1 partition - waits for NB2's comm from PREVIOUS segment
            partitions.append(partition_class(
                partition_id=partition_id,
                nano_batch_idx=0,
                comp_ops=comp_ops,
                comm=prev_comm,  # Wait for NB2's previous segment's comm
            ))
            partition_id += 1
            
            # Nanobatch 2 partition - waits for NB1's comm from CURRENT segment
            # (NB1 just started its comm after executing comp_ops above)
            partitions.append(partition_class(
                partition_id=partition_id,
                nano_batch_idx=1,
                comp_ops=comp_ops,
                comm=comm_after,  # Wait for NB1's current segment's comm
            ))
            partition_id += 1
            
            # Update prev_comm for next iteration
            # (NB2 will start comm_after after its compute, NB1 in next segment waits for it)
            prev_comm = comm_after
        
        return partitions
    
    def _split_by_communications(
        self,
        ops: List[Union[ComputeOp, CommunicationOp]],
    ) -> List[Tuple[List[ComputeOp], Optional[CommunicationOp]]]:
        """
        Split ops into segments by communication boundaries.
        
        Input: [A, AR, B, C, AR, D]
        Output: [([A], AR), ([B, C], AR), ([D], None)]
        """
        segments = []
        current_comp_ops = []
        
        for op in ops:
            if isinstance(op, ComputeOp):
                current_comp_ops.append(op)
            elif isinstance(op, CommunicationOp):
                if current_comp_ops:
                    segments.append((current_comp_ops, op))
                    current_comp_ops = []
                else:
                    # Communication at start - attach to previous segment or skip
                    if segments:
                        segments[-1] = (segments[-1][0], op)
        
        # Handle trailing compute ops
        if current_comp_ops:
            segments.append((current_comp_ops, None))
        
        return segments
```

---

### Objective 4: Separate Forward and Backward Partition Classes

**Key Design Decision: Forward and backward partitions are SEPARATE classes**

Forward and backward passes have different communication patterns, so they should be
different classes rather than forward/backward methods on one partition class.

**Files to Create:**
- `kareus/megatron/core/partitions/forward_partition.py` (NEW)
- `kareus/megatron/core/partitions/backward_partition.py` (NEW)
- `kareus/megatron/core/partitions/context_manager.py` (NEW)

**Design:**

```python
# kareus/megatron/core/partitions/context_manager.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch

@dataclass
class TensorStore:
    """
    Stores tensors by their auto-generated tensor IDs.
    
    Tensors are identified by IDs like "t_0", "t_1", etc., which are
    automatically assigned by the TensorGraphBuilder.
    """
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def set(self, tensor_id: str, tensor: torch.Tensor):
        """Store a tensor by its auto-generated ID"""
        self.tensors[tensor_id] = tensor
    
    def get(self, tensor_id: str) -> Optional[torch.Tensor]:
        """Get a tensor by its auto-generated ID"""
        return self.tensors.get(tensor_id)
    
    def get_by_ports(self, ports: List['TensorPort']) -> List[torch.Tensor]:
        """Get tensors for a list of ports"""
        return [self.tensors.get(p.tensor_id) for p in ports]
    
    def set_from_ports(self, ports: List['TensorPort'], tensors: List[torch.Tensor]):
        """Store tensors from a list of ports"""
        for port, tensor in zip(ports, tensors):
            if tensor is not None:
                self.tensors[port.tensor_id] = tensor
    
    def clear(self):
        """Release all stored tensors to free memory."""
        self.tensors.clear()


@dataclass
class NanoBatchContext:
    """
    Context for a single nano-batch across partitions.
    
    Uses AUTOMATIC TENSOR ROUTING via TensorStore:
    - Tensors are stored/retrieved by auto-generated IDs (t_0, t_1, ...)
    - No manual name matching needed between operators
    - The TensorGraph defines which tensors flow where
    
    op_contexts is keyed by op_id (int), which is a property of
    PartitionableOperator assigned once during _build_partitions Step 1.
    Since both forward and backward ComputeOps reference the same operator
    instance, they share the same op_id, so backward can retrieve the
    context saved during forward.
    """
    batch_idx: int
    
    # Tensor storage - tensors are stored by auto-generated IDs
    tensor_store: TensorStore = field(default_factory=TensorStore)
    
    # Operation contexts for backward, keyed by op_id (PartitionableOperator.op_id)
    # Forward: create_op_context(op.operator.op_id) → saves OperationContext
    # Backward: get_op_context(op.operator.op_id) → retrieves same OperationContext
    op_contexts: Dict[int, 'OperationContext'] = field(default_factory=dict)
    
    # Saved tensors for backward (flattened from all op_contexts)
    _saved_tensors: List[torch.Tensor] = field(default_factory=list)
    _saved_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    def create_op_context(self, op_id: int) -> 'OperationContext':
        """
        Create operation context for an operator (called during forward).
        
        Args:
            op_id: PartitionableOperator.op_id (assigned in _build_partitions Step 1)
        """
        from transformer_engine.pytorch.ops.op import OperationContext
        ctx = OperationContext()
        self.op_contexts[op_id] = ctx
        return ctx
    
    def get_op_context(self, op_id: int) -> 'OperationContext':
        """
        Get operation context for backward.
        
        Args:
            op_id: PartitionableOperator.op_id - same value used in create_op_context
                   since forward and backward ComputeOps share the same operator.
        """
        return self.op_contexts[op_id]
    
    def flatten_saved_tensors(self) -> List[torch.Tensor]:
        """Flatten all op contexts' saved tensors for autograd"""
        self._saved_tensors = []
        for op_id, ctx in self.op_contexts.items():
            range_start = len(self._saved_tensors)
            if ctx.to_save is not None:
                self._saved_tensors.extend(ctx.to_save)
            range_end = len(self._saved_tensors)
            ctx.to_save = None
            self._saved_ranges[op_id] = (range_start, range_end)
        return self._saved_tensors
    
    def restore_saved_tensors(self, saved_tensors: Tuple[torch.Tensor, ...]):
        """Restore saved tensors to op contexts for backward"""
        for op_id, ctx in self.op_contexts.items():
            if op_id in self._saved_ranges:
                ctx.saved_tensors = saved_tensors[slice(*self._saved_ranges[op_id])]


# kareus/megatron/core/partitions/forward_partition.py

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import torch

OverlapWindow = Tuple[int, int]  # (comm_start, comm_end) - fused_idx to launch/finish comm
ResourceShape = Tuple[int, int]  # (sm_num, block_size) - SM allocation for comm


@dataclass
class ForwardPartition:
    """
    Partition for forward pass execution.
    
    KEY FEATURES:
    1. AUTOMATIC TENSOR ROUTING via TensorGraph - tensors flow by auto-generated IDs
    2. SCHEDULER INTEGRATION - overlap_window and sm_configs loaded from scheduler
    3. GENERIC COMMUNICATION - handles 1-port (AllReduce) and 2-port (AllGatherKV) comms
    
    Each partition contains:
    - comp_ops: List of ComputeOp in this partition (for THIS nanobatch)
    - comm_op: Optional CommunicationOp (for the OTHER nanobatch)
    
    Tensor Routing (CROSS-NANOBATCH via ctx / pre_ctx):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  ctx      = NanoBatchContext for THIS nanobatch                     │
    │  pre_ctx  = NanoBatchContext for the OTHER nanobatch                │
    │                                                                      │
    │  COMPUTE OPS: read/write tensors via ctx.tensor_store                │
    │    input:  ctx.tensor_store.get(op.input_ports[i].tensor_id)         │
    │    output: ctx.tensor_store.set(op.output_ports[i].tensor_id, t)     │
    │                                                                      │
    │  COMM OP: read/write tensors via pre_ctx.tensor_store                │
    │    input:  pre_ctx.tensor_store.get(comm_op.input_ports[i].tensor_id)│
    │    output: pre_ctx.tensor_store.set(comm_op.output_ports[i].tensor_id, t)│
    │                                                                      │
    │  This means the comm op processes the OTHER nanobatch's data         │
    │  while compute ops process THIS nanobatch's data concurrently.       │
    └──────────────────────────────────────────────────────────────────────┘
    
    Comparison with current partition_fuser.py:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  OLD (partition_fuser.py):                                           │
    │    - Manual isinstance() to determine extra_inputs per op type       │
    │    - Hardcoded variables: x, bias, residual, key, value              │
    │    - comm_input passed as explicit function argument                  │
    │    - Manual extra_outputs extraction by op type                      │
    │                                                                      │
    │  NEW (ForwardPartition):                                             │
    │    - Port-based routing: op.input_ports / op.output_ports            │
    │    - TensorStore with auto-generated tensor_ids (t_0, t_1, ...)      │
    │    - comm reads/writes via pre_ctx.tensor_store                      │
    │    - No isinstance() checks, no manual variable matching             │
    └──────────────────────────────────────────────────────────────────────┘
    """
    partition_id: int
    partition_key: str = ""  # Assigned by _build_partitions after forming partitions
    nano_batch_idx: int  # 0 or 1
    comp_ops: List[ComputeOp] = field(default_factory=list)
    comm_op: Optional[CommunicationOp] = None  # operates on the OTHER nanobatch
    
    # Schedule config - loaded from scheduler at runtime via load_schedule()
    _schedule_config: Optional[Tuple[OverlapWindow, ResourceShape]] = None

    def load_schedule(self, schedule: 'ScheduleItem | ScheduleItemCP'):
        """
        Load overlap_window and sm_configs from scheduler's current_schedule.
        
        Called by transformer_block before executing this partition.
        Maps partition_key to its config from the schedule.
        """
        config = getattr(schedule, self.partition_key, None)
        if config is not None:
            self._schedule_config = (
                config.overlap_window,
                config.resource_shape,
            )
    
    def get_comm_config(self) -> Tuple[OverlapWindow, ResourceShape]:
        if self._schedule_config:
            return self._schedule_config
        return ((-1, -1), (None, None))

    def execute(
        self,
        ctx: NanoBatchContext,      # THIS nanobatch - for compute ops
        pre_ctx: NanoBatchContext,   # OTHER nanobatch - for comm op
    ):
        """
        Execute forward partition with AUTOMATIC TENSOR ROUTING.
        
        All tensor routing is done via tensor_id from TensorGraph ports:
        - Compute ops: ctx.tensor_store.get/set by tensor_id
        - Comm op: pre_ctx.tensor_store.get/set by tensor_id
        
        No isinstance() checks needed. No manual variable matching.
        
        Mirrors the pattern from partition_fuser.py but replaces manual
        type-based tensor juggling with port-based TensorStore lookups.
        
        Args:
            ctx: NanoBatchContext for this nanobatch (compute ops read/write here)
            pre_ctx: NanoBatchContext for the other nanobatch (comm op reads/writes here)
        """
        current_stream = torch.cuda.current_stream()
        is_grad_enabled = torch.is_grad_enabled()
        
        # --- Communication overlap setup ---
        # (same pattern as partition_fuser.py lines 110-120)
        if self.comm_op is not None:
            (comm_start, comm_end), (sm_num, block_size) = self.get_comm_config()
        else:
            comm_start, comm_end = -1, -1
            sm_num, block_size = None, None
        
        if comm_start == 0:
            self.comm_op.event_record(current_stream)
        
        # Track comm output tensors (captured at launch, valid after sync)
        # fuser_forward returns (main_output, extra_outputs_per_basic_op)
        # where extra_outputs_per_basic_op is List[Tuple[Tensor, ...]]
        comm_output = None
        comm_extra_outputs = []
        
        # --- Iterate compute ops ---
        # (replaces partition_fuser.py lines 128-205, removing isinstance checks)
        for fused_idx, op in enumerate(self.comp_ops):
            
            # 1. Create OperationContext for this op (for autograd save/restore)
            #    (same as partition_fuser.py line 107)
            #    Keyed by op.operator.op_id (assigned in _build_partitions Step 1).
            #    Backward retrieves with the same key since it shares the operator.
            op_ctx = ctx.create_op_context(op.operator.op_id)
            
            # 2. Get input tensors by port tensor_ids from THIS nanobatch's store
            #    OLD: manual isinstance() to figure out extra_inputs
            #    NEW: just read all input ports from tensor_store
            #
            #    Port 0 = main tensor (x / hidden_states)
            #    Port 1+ = extra inputs (bias, residual, key, value, rotary, etc.)
            x = ctx.tensor_store.get(op.input_ports[0].tensor_id)
            extra_inputs = []
            if len(op.input_ports) > 1:
                extra_inputs = [tuple(
                    ctx.tensor_store.get(p.tensor_id) for p in op.input_ports[1:]
                )]
            
            # 3. Track requires_grad
            #    (same as partition_fuser.py lines 147-158)
            requires_grad = is_grad_enabled and x.requires_grad
            if is_grad_enabled and not requires_grad:
                requires_grad = any(p.requires_grad for p in op.operator.parameters())
            if is_grad_enabled and not requires_grad:
                requires_grad = any(
                    t is not None and t.requires_grad
                    for xs in extra_inputs for t in xs
                )
            op_ctx.requires_grad = requires_grad
            if requires_grad != x.requires_grad:
                x = x.requires_grad_() if requires_grad else x.detach()
            
            # 4. Communication overlap: launch comm at overlap window
            #    (same as partition_fuser.py lines 166-174)
            #    IMPORTANT: comm op reads from pre_ctx (OTHER nanobatch)
            #    Multi-port comm ops (e.g. AllGatherKV) have 2+ input ports;
            #    port 0 is the main input, ports 1+ are passed as extra_inputs.
            if comm_start == fused_idx:
                self.comm_op.event_wait()
                comm_input = pre_ctx.tensor_store.get(
                    self.comm_op.input_ports[0].tensor_id
                )
                comm_extra_inputs = []
                if len(self.comm_op.input_ports) > 1:
                    comm_extra_inputs = [tuple(
                        pre_ctx.tensor_store.get(p.tensor_id)
                        for p in self.comm_op.input_ports[1:]
                    )]
                comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
                    [OperationContext()],
                    comm_input,
                    basic_op_extra_inputs=comm_extra_inputs,
                    basic_op_prev_ops=[None],
                    basic_op_next_ops=[None],
                    basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}],
                )
            
            # 5. Execute compute op
            #    (same as partition_fuser.py lines 176-183)
            x, fused_op_extra_outputs = op.operator.fuser_forward(
                [op_ctx],
                x,
                basic_op_extra_inputs=extra_inputs,
                basic_op_prev_ops=[None],
                basic_op_next_ops=[None],
                basic_op_kwargs=[{}],
            )
            
            # 6. Record event for comm overlap
            #    (same as partition_fuser.py lines 186-187)
            if fused_idx == comm_start - 1:
                self.comm_op.event_record(current_stream)
            
            # 7. Store output tensors by port tensor_ids to THIS nanobatch's store
            #    OLD: manual isinstance() to extract key, value, bias, etc.
            #    NEW: just write all output ports to tensor_store
            #
            #    Port 0 = main output (x)
            #    Port 1+ = extra outputs (key, value, bias, etc.)
            x.requires_grad_(requires_grad=requires_grad)
            ctx.tensor_store.set(op.output_ports[0].tensor_id, x)
            for port_idx, port in enumerate(op.output_ports[1:]):
                extra_out = fused_op_extra_outputs[0][port_idx] if fused_op_extra_outputs else None
                if extra_out is not None:
                    extra_out.requires_grad_(requires_grad=requires_grad)
                    ctx.tensor_store.set(port.tensor_id, extra_out)
        
        # --- Handle non-overlapped comm ---
        # (same as partition_fuser.py lines 207-215)
        if comm_start == -1 and self.comm_op is not None:
            self.comm_op.event_record(current_stream)
            self.comm_op.event_wait()
            comm_input = pre_ctx.tensor_store.get(
                self.comm_op.input_ports[0].tensor_id
            )
            comm_extra_inputs = []
            if len(self.comm_op.input_ports) > 1:
                comm_extra_inputs = [tuple(
                    pre_ctx.tensor_store.get(p.tensor_id)
                    for p in self.comm_op.input_ports[1:]
                )]
            comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
                [OperationContext()],
                comm_input,
                basic_op_extra_inputs=comm_extra_inputs,
                basic_op_prev_ops=[None],
                basic_op_next_ops=[None],
                basic_op_kwargs=[{"sm_num": sm_num, "block_size": block_size}],
            )
        
        # --- Sync comm and store comm output to OTHER nanobatch's store ---
        # (same as partition_fuser.py line 253)
        if self.comm_op is not None:
            self.comm_op.sync(current_stream)
            # Write comm outputs to pre_ctx (OTHER nanobatch)
            # The other nanobatch's next partition will read these.
            # comm_output/comm_extra_outputs were captured at fuser_forward call;
            # tensors are pre-allocated buffers filled in-place by the async comm op.
            pre_ctx.tensor_store.set(
                self.comm_op.output_ports[0].tensor_id, comm_output
            )
            if comm_extra_outputs:
                for port_idx, port in enumerate(self.comm_op.output_ports[1:]):
                    extra_out = comm_extra_outputs[0][port_idx]
                    if extra_out is not None:
                        pre_ctx.tensor_store.set(port.tensor_id, extra_out)
            
            
    
# kareus/megatron/core/partitions/backward_partition.py

@dataclass
class BackwardPartition:
    """
    Partition for backward pass execution.
    
    Uses a SEPARATE backward TensorGraph where:
    1. Ops are in REVERSE order (last layer first)
    2. Channel semantics are REVERSED via ComputeOpSpec(is_backward=True):
       - input_channels  = [grad_{ch} for ch in forward output_channels]
       - output_channels = [grad_{ch} for ch in forward input_channels]
    3. ResidualForkOp backward auto-derives: input [grad_main, grad_residual] → output [grad_main]
       Its fuser_backward performs the accumulation: grad_input = grad_main + grad_residual
    4. Communication patterns differ from forward (e.g., backward may have
       AllReduce where forward didn't, and vice versa)
    
    Channel reversal example (asymmetric operator like QKVPostProcess):
    ┌──────────────────────────────────────────────────────────────────┐
    │  Forward:   input [Channel(0,"main")]                            │
    │             → output [Channel(0,"main"), Channel(1,"key"),       │
    │                       Channel(2,"value")]                        │
    │  Backward:  input [Channel(0,"grad_main"), Channel(1,"grad_key"),│
    │                    Channel(2,"grad_value")]  (auto-derived)      │
    │             → output [Channel(0,"grad_main")]  (auto-derived)    │
    │                                                                  │
    │  ComputeOpSpec(is_backward=True) auto-derives backward channels  │
    │  from forward declarations — no explicit backward needed.        │
    └──────────────────────────────────────────────────────────────────┘
    
    Tensor Routing (same ctx/pre_ctx pattern as ForwardPartition):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  COMPUTE OPS: read/write GRAD tensors via ctx.tensor_store           │
    │    grad_input:  ctx.tensor_store.get(op.input_ports[i].tensor_id)    │
    │    grad_output: ctx.tensor_store.set(op.output_ports[i].tensor_id, g)│
    │                                                                      │
    │  COMM OP: read/write GRAD tensors via pre_ctx.tensor_store           │
    │    input:  pre_ctx.tensor_store.get(comm_op.input_ports[i].tensor_id)│
    │    output: pre_ctx.tensor_store.set(comm_op.output_ports[i].tensor_id, g)│
    │                                                                      │
    │  OP CONTEXTS: retrieved from ctx (saved during forward pass)         │
    │    op_ctx = ctx.get_op_context(op.op_id) → has .saved_tensors        │
    └──────────────────────────────────────────────────────────────────────┘
    
    Comparison with current partition_fuser.py backward:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  OLD (partition_fuser.py backward, lines 314-377):                   │
    │    - dx = grad_output, then manual isinstance() for grad_extra       │
    │    - isinstance(op, QKVPostProcessOp) → grad_key, grad_value         │
    │    - isinstance(op, BiasDropoutAddOp) → grad_bias, grad_residual     │
    │    - isinstance(op, LayerNorm) → dx = dx + grad_residual             │
    │    - Manual variable juggling between ops                            │
    │                                                                      │
    │  NEW (BackwardPartition):                                            │
    │    - All grad tensors routed via tensor_id in backward TensorGraph   │
    │    - op.input_ports → read grad tensors from ctx.tensor_store        │
    │    - op.output_ports → write grad tensors to ctx.tensor_store        │
    │    - No isinstance() checks, no manual grad variable matching        │
    └──────────────────────────────────────────────────────────────────────┘
    """
    partition_id: int
    partition_key: str = ""  # Assigned by _build_partitions after forming partitions
    nano_batch_idx: int  # 0 or 1
    comp_ops: List[ComputeOp] = field(default_factory=list)
    comm_op: Optional[CommunicationOp] = None  # operates on the OTHER nanobatch
    
    # Schedule config - same structure as ForwardPartition
    _schedule_config: Optional[Tuple[OverlapWindow, ResourceShape]] = None
    
    def load_schedule(self, schedule: 'ScheduleItem | ScheduleItemCP'):
        """Load overlap_window and sm_configs from scheduler's current_schedule."""
        config = getattr(schedule, self.partition_key, None)
        if config is not None:
            self._schedule_config = (
                config.overlap_window,
                config.resource_shape,
            )
    
    def get_comm_config(self) -> Tuple[OverlapWindow, ResourceShape]:
        if self._schedule_config:
            return self._schedule_config
        return ((-1, -1), (None, None))
    
    def execute(
        self,
        ctx: NanoBatchContext,      # THIS nanobatch - for compute ops + op_contexts
        pre_ctx: NanoBatchContext,   # OTHER nanobatch - for comm op
    ):
        """
        Execute backward partition with AUTOMATIC TENSOR ROUTING.
        
        Mirrors ForwardPartition.execute() but for gradients:
        - Compute ops: call op.operator.fuser_backward() with saved OperationContext
        - Comm op: processes OTHER nanobatch's gradients via pre_ctx
        - All tensor routing via tensor_id (no isinstance checks)
        
        Backward TensorGraph ports are REVERSED relative to forward:
        - input_ports → gradient of forward output (what we receive)
        - output_ports → gradient of forward input (what we produce)
        
        The OperationContext (with saved_tensors from forward) is retrieved
        from ctx.get_op_context(op.op_id), where op_id matches the forward
        ComputeOp that created the context.
        
        Args:
            ctx: NanoBatchContext for this nanobatch
                 - tensor_store: holds grad tensors flowing through backward graph
                 - op_contexts: saved from forward, contains saved_tensors
            pre_ctx: NanoBatchContext for the other nanobatch
                 - tensor_store: comm op reads/writes grad tensors here
        """
        current_stream = torch.cuda.current_stream()
        
        # --- Communication overlap setup ---
        # (same pattern as partition_fuser.py backward lines 295-302)
        if self.comm_op is not None:
            (comm_start, comm_end), (sm_num, block_size) = self.get_comm_config()
        else:
            comm_start, comm_end = -1, -1
            sm_num, block_size = None, None
        
        if comm_start == 0:
            self.comm_op.event_record(current_stream)
        
        # Track comm output tensors (captured at launch, valid after sync)
        comm_output = None
        comm_extra_outputs = []
        
        # --- Collect parameter gradients ---
        # (same as partition_fuser.py lines 309, 398-411)
        grad_params = {}  # op_id -> list of param grads
        
        # --- Iterate compute ops ---
        # (replaces partition_fuser.py lines 314-377, removing isinstance checks)
        for fused_idx, op in enumerate(self.comp_ops):
            
            # 1. Retrieve OperationContext saved during forward pass
            #    Contains saved_tensors needed for backward computation.
            #    Keyed by op.operator.op_id (same value used in forward's
            #    create_op_context, since forward and backward ComputeOps
            #    share the same operator instance).
            op_ctx = ctx.get_op_context(op.operator.op_id)
            
            # Stop if no more gradients are required
            # (same as partition_fuser.py lines 317-319)
            if not op_ctx.requires_grad:
                break
            
            # 2. Get grad input tensors by port tensor_ids from THIS nanobatch
            #    OLD: manual isinstance() to determine grad_extra_outputs per op type
            #         e.g. isinstance(op, QKVPostProcessOp) → (grad_key, grad_value)
            #    NEW: just read all input ports from tensor_store
            #
            #    Backward input port 0 = grad of forward output (dx / grad_output)
            #    Backward input port 1+ = grad of extra forward outputs
            #                             (grad_key, grad_value, grad_bias, etc.)
            dx = ctx.tensor_store.get(op.input_ports[0].tensor_id)
            grad_extra_outputs = []
            if len(op.input_ports) > 1:
                grad_extra_outputs = [tuple(
                    ctx.tensor_store.get(p.tensor_id) for p in op.input_ports[1:]
                )]
            
            # 3. Communication overlap: launch backward comm at overlap window
            #    (same as partition_fuser.py lines 330-343)
            #    IMPORTANT: backward comm op also uses pre_ctx (OTHER nanobatch)
            #    Multi-port comm ops (e.g. ReduceScatterKV) have 2+ input ports;
            #    port 0 is the main input, ports 1+ are passed as extra_inputs.
            if comm_start == fused_idx:
                self.comm_op.event_wait()
                grad_comm_input = pre_ctx.tensor_store.get(
                    self.comm_op.input_ports[0].tensor_id
                )
                grad_comm_extra_inputs = []
                if len(self.comm_op.input_ports) > 1:
                    grad_comm_extra_inputs = [tuple(
                        pre_ctx.tensor_store.get(p.tensor_id)
                        for p in self.comm_op.input_ports[1:]
                    )]
                comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
                    [None],
                    grad_comm_input,
                    basic_op_extra_inputs=grad_comm_extra_inputs,
                    basic_op_prev_ops=[None],
                    basic_op_next_ops=[None],
                    basic_op_kwargs=[{
                        "sm_num": sm_num,
                        "block_size": block_size,
                        "backward": True,
                    }],
                )
            
            # 4. Execute backward compute op
            #    (same as partition_fuser.py lines 346-353)
            dx, fused_op_grad_params, fused_op_grad_extra_inputs = op.operator.fuser_backward(
                [op_ctx],
                dx,
                basic_op_grad_extra_outputs=grad_extra_outputs,
            )
            
            # Store parameter gradients
            # fused_op_grad_params is List[Tuple[Tensor, ...]], flatten to List[Tensor]
            grad_params[op.operator.op_id] = [
                t for param_tuple in fused_op_grad_params for t in param_tuple
            ]
            op_ctx.saved_tensors = None  # Free saved tensors
            
            # 5. Record event for comm overlap
            #    (same as partition_fuser.py lines 355-356)
            if fused_idx == comm_start - 1:
                self.comm_op.event_record(current_stream)
            
            # 6. Store grad output tensors by port tensor_ids to THIS nanobatch
            #    OLD: manual isinstance() to extract grad_bias, grad_residual, etc.
            #         e.g. isinstance(op, BiasDropoutAddOp) → grad_bias, grad_residual
            #         e.g. isinstance(op, LayerNorm) → dx = dx + grad_residual
            #    NEW: just write all output ports to tensor_store
            #
            #    Backward output port 0 = grad of forward input (dx)
            #    Backward output port 1+ = grad of extra forward inputs
            #                              (grad_bias, grad_residual, etc.)
            ctx.tensor_store.set(op.output_ports[0].tensor_id, dx)
            if fused_op_grad_extra_inputs:
                for port_idx, port in enumerate(op.output_ports[1:]):
                    grad_extra = fused_op_grad_extra_inputs[0][port_idx] \
                        if fused_op_grad_extra_inputs[0] else None
                    if grad_extra is not None:
                        ctx.tensor_store.set(port.tensor_id, grad_extra)
        
        # --- Handle non-overlapped backward comm ---
        # (same as partition_fuser.py lines 378-392)
        if comm_start == -1 and self.comm_op is not None:
            self.comm_op.event_record(current_stream)
            self.comm_op.event_wait()
            grad_comm_input = pre_ctx.tensor_store.get(
                self.comm_op.input_ports[0].tensor_id
            )
            grad_comm_extra_inputs = []
            if len(self.comm_op.input_ports) > 1:
                grad_comm_extra_inputs = [tuple(
                    pre_ctx.tensor_store.get(p.tensor_id)
                    for p in self.comm_op.input_ports[1:]
                )]
            comm_output, comm_extra_outputs = self.comm_op.fuser_forward(
                [None],
                grad_comm_input,
                basic_op_extra_inputs=grad_comm_extra_inputs,
                basic_op_prev_ops=[None],
                basic_op_next_ops=[None],
                basic_op_kwargs=[{
                    "sm_num": sm_num,
                    "block_size": block_size,
                    "backward": True,
                }],
            )
        
        # --- Sync backward comm and store output to OTHER nanobatch ---
        # (same as partition_fuser.py line 421)
        if self.comm_op is not None:
            self.comm_op.sync(current_stream)
            # Write comm grad outputs to pre_ctx (OTHER nanobatch).
            # comm_output/comm_extra_outputs were captured at fuser_forward call;
            # tensors are pre-allocated buffers filled in-place by the async comm op.
            pre_ctx.tensor_store.set(
                self.comm_op.output_ports[0].tensor_id, comm_output
            )
            if comm_extra_outputs:
                for port_idx, port in enumerate(self.comm_op.output_ports[1:]):
                    extra_out = comm_extra_outputs[0][port_idx]
                    if extra_out is not None:
                        pre_ctx.tensor_store.set(port.tensor_id, extra_out)
        
        return grad_params

```

---

### Objective 5: Unified Autograd Function with Scheduler Integration

**Files to Create/Modify:**
- `kareus/megatron/core/partitions/autograd_function.py` (NEW)

**Key Features:**
1. AUTOMATIC TENSOR ROUTING via TensorGraph - tensors identified by auto-generated IDs
2. SCHEDULER INTEGRATION - loads overlap_window/sm_configs from scheduler.current_schedule
3. SINGLE AUTOGRAD BOUNDARY for the entire transformer block (replaces per-fuser autograd functions)

**Why a single autograd function?**

The current architecture has separate autograd functions per fuser type (`_PartitionFuserAutogradFunction`,
`_QKVFuserAutogradFunction`, `_AttnOprojFuserAutogradFunction`). Each manages its own `save_for_backward`
and gradient computation. This is fragile because:
- Cross-partition tensor flow (bias, residual) requires manual plumbing between fusers
- Parameter gradients must be manually accumulated across fusers
- The TransformerBlock forward loop must manually manage residual/comm state variables

A unified autograd function wraps the ENTIRE block execution (all layers, all partitions):
- Forward: executes all ForwardPartitions in interleaved order
- Backward: executes all BackwardPartitions in interleaved order
- All tensor routing happens automatically via TensorStore + tensor_ids
- Parameter gradients are accumulated naturally across partitions

**Design:**

```python
# kareus/megatron/core/partitions/autograd_function.py

from typing import Dict, List, Optional, Tuple, Any
import torch

class TransformerBlockAutogradFunction(torch.autograd.Function):
    """
    Unified autograd function for the entire transformer block.
    
    Wraps ALL forward/backward partitions across ALL layers in a single
    autograd boundary. This replaces the per-fuser autograd functions
    (_PartitionFuserAutogradFunction, _QKVFuserAutogradFunction, etc.).
    
    KEY DESIGN DECISIONS:
    
    1. TWO NanoBatchContexts (ctx_nb1, ctx_nb2):
       Each has its own TensorStore for tensor routing and op_contexts
       for saving/restoring OperationContext across forward and backward.
       Forward partitions write to ctx, backward partitions read from ctx.
    
    2. ctx vs pre_ctx PATTERN:
       Each partition operates on ONE nanobatch's compute (ctx) while
       overlapping with the OTHER nanobatch's communication (pre_ctx).
       - NB1 partition: compute reads/writes ctx_nb1, comm reads/writes ctx_nb2
       - NB2 partition: compute reads/writes ctx_nb2, comm reads/writes ctx_nb1
    
    3. PARAMETER GRADIENTS:
       All parameters are passed via *params so autograd tracks them.
       Backward partitions return grad_params keyed by op_id.
       Gradients from both nanobatches are summed for each parameter.
    
    4. FINAL TENSOR LOOKUP:
       The forward/backward TensorGraph's channel_registry provides the
       final tensor_id for "main" / "grad_main" channels. No need to
       search by highest index — just use forward_graph.get_output_channel("main").
    
    Comparison with current architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  OLD: TransformerBlock.forward() loop                               │
    │    for layer in layers:                                             │
    │      h1, res1, comm1 = layer.forward_attention(batch_idx=1, ...)    │
    │      h2, res2, comm2 = layer.forward_attention(batch_idx=2, ...)    │
    │      h1, res1, comm1 = layer.forward_mlp(batch_idx=1, ...)          │
    │      h2, res2, comm2 = layer.forward_mlp(batch_idx=2, ...)          │
    │    # Manual state threading: h1, h2, res1, res2, comm1, comm2       │
    │    # Each fuser has its own autograd function                        │
    │                                                                      │
    │  NEW: TransformerBlockAutogradFunction.apply()                       │
    │    for partition in forward_partitions:                               │
    │      partition.execute(ctx=ctx_nbX, pre_ctx=ctx_nbY)                 │
    │    # All state in TensorStores, single autograd boundary             │
    └──────────────────────────────────────────────────────────────────────┘
    """
    
    @staticmethod
    def forward(
        func_ctx,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        forward_partitions: List['ForwardPartition'],
        backward_partitions: List['BackwardPartition'],
        forward_tensor_graph: 'TensorGraph',   # For final tensor_id lookup
        backward_tensor_graph: 'TensorGraph',   # For final grad tensor_id lookup
        scheduler: Optional['PipelineCommScheduler'],
        config: Any,
        *params,  # All parameters from all layers (for autograd tracking)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute all forward partitions with scheduler-loaded configs.
        
        Steps:
        1. Create NanoBatchContexts and seed TensorStores
        2. Load schedule from scheduler.current_schedule
        3. Execute partitions in interleaved NB1/NB2 order
        4. Extract final hidden_states from TensorStores
        5. Save contexts for backward
        
        Args:
            func_ctx: PyTorch autograd function context (for save_for_backward)
            hidden_states_1: Input tensor for nano-batch 1 [s, b/2, h]
            hidden_states_2: Input tensor for nano-batch 2 [s, b/2, h]
            rotary_pos_emb: Rotary positional embeddings (shared across NB1/NB2)
            attention_mask: Attention mask (shared across NB1/NB2)
            forward_partitions: List of ForwardPartition (interleaved NB1/NB2 order)
            backward_partitions: List of BackwardPartition (saved for backward)
            forward_tensor_graph: Forward TensorGraph (for final output tensor_id)
            backward_tensor_graph: Backward TensorGraph (saved for backward)
            scheduler: PipelineCommScheduler with current_schedule
            config: TransformerConfig
            *params: All parameters from all layers
        
        Returns:
            (h1_out, h2_out): Output tensors for both nano-batches
        """
        is_grad_enabled = torch.is_grad_enabled()
        
        # --- Step 1: Create NanoBatchContexts and seed TensorStores ---
        ctx_nb1 = NanoBatchContext(batch_idx=0)
        ctx_nb2 = NanoBatchContext(batch_idx=1)
        
        # Seed initial tensors with the tensor_ids from the forward graph
        ctx_nb1.tensor_store.set("t_input_0", hidden_states_1)
        ctx_nb2.tensor_store.set("t_input_0", hidden_states_2)
        
        # Seed external inputs (shared across both nano-batches and all layers)
        # rotary_pos_emb persists in the channel registry because no operator
        # overwrites it — every RotaryEmbeddingOp reads the same tensor_id.
        if rotary_pos_emb is not None:
            ctx_nb1.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)
            ctx_nb2.tensor_store.set("t_rotary_pos_emb", rotary_pos_emb)
        
        # --- Step 2: Load schedule ---
        current_schedule = scheduler.current_schedule if scheduler else None
        
        # --- Step 3: Execute forward partitions in interleaved order ---
        # Partitions are already ordered: [NB1_seg0, NB2_seg0, NB1_seg1, NB2_seg1, ...]
        # Each partition executes compute on its own NB and comm on the other NB.
        for partition in forward_partitions:
            # Load overlap_window and sm_configs from scheduler
            if current_schedule:
                partition.load_schedule(current_schedule)
            
            if partition.nano_batch_idx == 0:
                partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
            else:
                partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
        
        # --- Step 4: Extract final outputs ---
        # The forward TensorGraph's channel_registry tells us the final
        # tensor_id for the "main" channel (last operator's output).
        final_tensor_id = forward_tensor_graph.get_output_channel("main")
        h1_out = ctx_nb1.tensor_store.get(final_tensor_id)
        h2_out = ctx_nb2.tensor_store.get(final_tensor_id)
        
        # --- Step 5: Save for backward ---
        if is_grad_enabled:
            # Save non-tensor state on the autograd context
            func_ctx.backward_partitions = backward_partitions
            func_ctx.backward_tensor_graph = backward_tensor_graph
            func_ctx.scheduler = scheduler
            func_ctx.nano_ctx_1 = ctx_nb1
            func_ctx.nano_ctx_2 = ctx_nb2
            func_ctx.num_params = len(params)
            
            # Flatten saved tensors from all op_contexts in both NanoBatchContexts.
            # Each OperationContext.to_save contains tensors needed for backward
            # (e.g., input activations, masks, scaling factors).
            # We flatten them into a single list and save via PyTorch's mechanism.
            saved_1 = ctx_nb1.flatten_saved_tensors()
            saved_2 = ctx_nb2.flatten_saved_tensors()
            func_ctx.num_saved_1 = len(saved_1)
            func_ctx.save_for_backward(*saved_1, *saved_2)
        
        return h1_out, h2_out
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        func_ctx,
        grad_h1: torch.Tensor,
        grad_h2: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Execute all backward partitions with scheduler-loaded configs.
        
        Steps:
        1. Restore saved tensors to NanoBatchContexts
        2. Seed gradient TensorStores with grad_output
        3. Execute backward partitions in interleaved order
        4. Collect and merge parameter gradients
        5. Return gradients for all inputs
        
        Args:
            func_ctx: PyTorch autograd function context (with saved tensors)
            grad_h1: Gradient of loss w.r.t. h1_out [s, b/2, h]
            grad_h2: Gradient of loss w.r.t. h2_out [s, b/2, h]
        
        Returns:
            Tuple of gradients matching forward() signature:
            (grad_h1, grad_h2, None, None, None, None, None, None, None, None,
             *grad_params)
        """
        backward_partitions = func_ctx.backward_partitions
        backward_tensor_graph = func_ctx.backward_tensor_graph
        scheduler = func_ctx.scheduler
        ctx_nb1 = func_ctx.nano_ctx_1
        ctx_nb2 = func_ctx.nano_ctx_2
        num_params = func_ctx.num_params
        
        # --- Step 1: Restore saved tensors ---
        # Split saved_tensors back into NB1 and NB2 portions,
        # then restore into each NanoBatchContext's op_contexts.
        saved_tensors = func_ctx.saved_tensors
        num_saved_1 = func_ctx.num_saved_1
        ctx_nb1.restore_saved_tensors(saved_tensors[:num_saved_1])
        ctx_nb2.restore_saved_tensors(saved_tensors[num_saved_1:])
        
        # Free forward tensors that are no longer needed.
        # Forward tensor_ids (t_0, t_1, ...) don't collide with backward
        # tensor_ids (t_grad_output_0, ...), but keeping them wastes memory.
        # Saved tensors needed for backward are already stored in op_contexts.
        ctx_nb1.tensor_store.clear()
        ctx_nb2.tensor_store.clear()
        
        # --- Step 2: Seed backward TensorStores with grad_output ---
        # The backward graph's initial channel "grad_main" maps to "t_grad_output_0"
        ctx_nb1.tensor_store.set("t_grad_output_0", grad_h1)
        ctx_nb2.tensor_store.set("t_grad_output_0", grad_h2)
        
        # --- Step 3: Load schedule ---
        current_schedule = scheduler.current_schedule if scheduler else None
        
        # --- Step 4: Execute backward partitions ---
        # Partitions are ordered for interleaved execution,
        # mirroring the forward pattern but with reversed ops.
        # Each partition returns grad_params: Dict[int, List] keyed by op_id.
        all_grad_params_nb1 = {}  # op_id → list of param grads
        all_grad_params_nb2 = {}
        
        for partition in backward_partitions:
            if current_schedule:
                partition.load_schedule(current_schedule)
            
            if partition.nano_batch_idx == 0:
                grad_params = partition.execute(ctx=ctx_nb1, pre_ctx=ctx_nb2)
            else:
                grad_params = partition.execute(ctx=ctx_nb2, pre_ctx=ctx_nb1)
            
            # Accumulate parameter gradients per nano-batch
            target = all_grad_params_nb1 if partition.nano_batch_idx == 0 else all_grad_params_nb2
            if grad_params:
                target.update(grad_params)
        
        # --- Step 5: Extract final input gradients ---
        # The backward graph's channel_registry tells us the final tensor_id
        # for "grad_main" (gradient of the original input hidden_states).
        final_grad_tensor_id = backward_tensor_graph.get_output_channel("grad_main")
        dh1 = ctx_nb1.tensor_store.get(final_grad_tensor_id)
        dh2 = ctx_nb2.tensor_store.get(final_grad_tensor_id)
        
        # --- Step 6: Merge parameter gradients from both nano-batches ---
        # Both nano-batches compute gradients for the SAME parameters
        # (shared weights). We sum them to get the total gradient.
        # 
        # grad_params dicts are keyed by op_id → list of param grads.
        # We need to produce a flat list matching the *params order in forward().
        combined_grad_params = _combine_param_grads(
            all_grad_params_nb1, all_grad_params_nb2, num_params
        )
        
        # Return gradients matching forward() signature:
        # (h1, h2, rotary_pos_emb, attention_mask,
        #  forward_partitions, backward_partitions,
        #  forward_tensor_graph, backward_tensor_graph,
        #  scheduler, config, *params)
        return (
            dh1,    # grad_hidden_states_1
            dh2,    # grad_hidden_states_2
            None,   # rotary_pos_emb (non-differentiable external input)
            None,   # attention_mask
            None,   # forward_partitions
            None,   # backward_partitions
            None,   # forward_tensor_graph
            None,   # backward_tensor_graph
            None,   # scheduler
            None,   # config
            *combined_grad_params,  # One gradient per parameter
        )


def _combine_param_grads(
    grad_params_nb1: Dict[int, List[Optional[torch.Tensor]]],
    grad_params_nb2: Dict[int, List[Optional[torch.Tensor]]],
    num_params: int,
) -> List[Optional[torch.Tensor]]:
    """
    Combine parameter gradients from both nano-batches.
    
    Both nano-batches share the same model parameters. Each backward
    partition produces gradients for its operators' parameters, keyed
    by op_id. We need to:
    1. Align gradients to the flat parameter list (matching *params order)
    2. Sum gradients from NB1 and NB2 for each parameter
    
    Args:
        grad_params_nb1: {op_id: [grad_param_0, grad_param_1, ...]} from NB1
        grad_params_nb2: {op_id: [grad_param_0, grad_param_1, ...]} from NB2
        num_params: Total number of parameters (len of *params in forward)
    
    Returns:
        List of gradients, one per parameter, in the same order as *params
    """
    # Build a flat list of gradients matching the parameter order.
    # Parameters are ordered by layer → operator → param within operator.
    # The op_id ordering matches this because op_ids are assigned sequentially
    # in _build_partitions Step 1 (iterating layers → operators in order).
    combined = [None] * num_params
    
    # Merge NB1 and NB2 gradients
    all_op_ids = sorted(set(grad_params_nb1.keys()) | set(grad_params_nb2.keys()))
    
    param_offset = 0
    for op_id in all_op_ids:
        grads_1 = grad_params_nb1.get(op_id, [])
        grads_2 = grad_params_nb2.get(op_id, [])
        num_op_params = max(len(grads_1), len(grads_2))
        
        for i in range(num_op_params):
            g1 = grads_1[i] if i < len(grads_1) else None
            g2 = grads_2[i] if i < len(grads_2) else None
            
            if g1 is not None and g2 is not None:
                combined[param_offset + i] = g1.add_(g2)  # In-place sum
            elif g1 is not None:
                combined[param_offset + i] = g1
            elif g2 is not None:
                combined[param_offset + i] = g2
        
        param_offset += num_op_params
    
    return combined
```

**Updated TransformerBlock.forward() call site:**

```python
# kareus/megatron/core/transformer/transformer_block.py (relevant section)
# Updates to the forward() method shown in Objective 1:

    def forward(self, ...):
        ...
        # Execute via autograd function with scheduler
        h1_out, h2_out = TransformerBlockAutogradFunction.apply(
            h1, h2,
            rotary_pos_emb,
            attention_mask,
            self.forward_partitions,
            self.backward_partitions,
            self.forward_tensor_graph,    # NEW: pass graph for final tensor_id lookup
            self.backward_tensor_graph,   # NEW: pass graph for final grad tensor_id lookup
            self.scheduler,
            self.config,
            *self._get_all_params(),
        )
        ...
```

**Updated `_build_partitions` to store tensor graphs:**

```python
    def _build_partitions(self):
        ...
        # Store tensor graphs for use by autograd function
        self.forward_tensor_graph = forward_tensor_graph
        self.backward_tensor_graph = backward_tensor_graph
        
        # Form partitions
        builder = PartitionBuilder(...)
        self.forward_partitions = builder.build_forward_partitions()
        self.backward_partitions = builder.build_backward_partitions()
```

---

### Objective 6: Mapping Current Fusers to New Design

**Current Manual Implementation (to be replaced):**
- `partition_fuser.py`: Handles Attention/MLP partitions (BDA→LN→compute→AR)
- `qkv_fuser.py`: Handles QKV partition with AllReduce
- `qkv_fuser2.py`: Handles QKV partition with AllGather (CP)
- `attn_oproj_fuser.py`: Handles Attention+OProj with complex CP communication

**Problems with current approach:**
1. Manual `isinstance()` checks for extra inputs/outputs
2. Duplicated code across fusers
3. Hardcoded communication patterns
4. Manual return of bias, residual, key, value, etc.
5. Manual tensor name matching between operators - prone to mismatch errors

**New Automatic Design using PORT-BASED I/O:**

Instead of manually declaring tensor names (which can mismatch between operators),
we use **numbered ports** and let the TensorGraphBuilder automatically:
1. Assign unique tensor IDs to each port connection
2. Connect output ports to input ports of subsequent operators
3. Handle multi-port communications (AllGatherKV with 2 ports)

