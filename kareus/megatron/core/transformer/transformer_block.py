# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from typing import List, Optional, Union
import json
import os
from dataclasses import asdict

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor

# Import dependencies from the original transformer_block.py
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    _get_block_submodules,
)

# Import the attention and MLP layers
from kareus.megatron.core.transformer.attention_layer import AttentionLayer
from kareus.megatron.core.transformer.mlp_layer import MLPLayer
# from kareus.megatron.core.transformer.mlp_output_layer import MLPOutputLayer

# Import the partition function
from kareus.megatron.core.transformer.partition_transformer_layer import (
    create_attention_and_mlp_layers_from_transformer_submodules,
    create_attention_and_mlp_layers_from_module_spec,
)

from kareus.utils.debug import save_tensors


try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None

    try:
        import apex  # pylint: disable=unused-import

        LayerNormImpl = FusedLayerNorm

    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        LayerNormImpl = WrappedTorchNorm


class CombinedLayerWrapper:
    """
    Wrapper class that combines AttentionLayer and MLPLayer to maintain
    backward compatibility with pipeline parallel code that expects a single layer.
    """
    
    def __init__(self, attention_layer, mlp_layer):
        self.attention_layer = attention_layer
        self.mlp_layer = mlp_layer
        # Expose layer_number for pipeline parallel compatibility
        self.layer_number = attention_layer.layer_number
    
    def __call__(self, *args, **kwargs):
        """Forward pass through both attention and MLP layers."""
        # Forward through attention layer
        pre_mlp_layernorm_output, residual, context = self.attention_layer(*args, **kwargs)
        
        # Forward through MLP layer
        hidden_states = self.mlp_layer(pre_mlp_layernorm_output, residual)
        
        return hidden_states, context
    
    def __getattr__(self, name):
        """Delegate attribute access to attention layer first, then MLP layer."""
        if hasattr(self.attention_layer, name):
            return getattr(self.attention_layer, name)
        elif hasattr(self.mlp_layer, name):
            return getattr(self.mlp_layer, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TransformerBlock(MegatronModule):
    """Transformer class."""

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

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

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
            assert (
                self.config.cpu_offloading is False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        self._init_layer_bda()
        # self._init_layer_tensor_parallel_comm()
    
    def set_tensor_parallel_group(self, tp_group: Optional[torch.distributed.ProcessGroup]=None) -> None:
        self._init_layer_tensor_parallel_comm()

    def _build_layers(self):
        # Build separate attention and MLP layers instead of combined transformer layers
        def build_attention_and_mlp_layers(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            fp8_init_context = get_fp8_context(layer_config, global_layer_number - 1, is_init=True)
            
            # Handle both ModuleSpec and TransformerLayerSubmodules
            if isinstance(layer_spec, ModuleSpec):
                attention_submodules, mlp_submodules = create_attention_and_mlp_layers_from_module_spec(layer_spec)
            else:
                # layer_spec is TransformerLayerSubmodules
                attention_submodules, mlp_submodules = create_attention_and_mlp_layers_from_transformer_submodules(layer_spec)
            
            with fp8_init_context:
                attention_layer = AttentionLayer(
                    config=layer_config, 
                    submodules=attention_submodules, 
                    layer_number=layer_number
                )
                mlp_layer = MLPLayer(
                    config=layer_config, 
                    submodules=mlp_submodules, 
                    layer_number=layer_number
                )
            return attention_layer, mlp_layer

        # Build separate attention and MLP layers
        attention_layers = []
        mlp_layers = []
        
        for i, layer_spec in enumerate(self.submodules.layer_specs):
            attention_layer, mlp_layer = build_attention_and_mlp_layers(layer_spec, i + 1)
            attention_layers.append(attention_layer)
            mlp_layers.append(mlp_layer)

        self.attention_layers = torch.nn.ModuleList(attention_layers)
        self.mlp_layers = torch.nn.ModuleList(mlp_layers)

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity
    
    def _init_layer_bda(self):
        mlp_bda = None
        for l_no in range(len(self.attention_layers)):
            attention_layer = self.attention_layers[l_no]
            mlp_layer = self.mlp_layers[l_no]

            attention_layer.prev_mlp_bda = mlp_bda

            mlp_layer.prev_self_attn_bda = attention_layer.post_self_attn_bda

            mlp_bda = mlp_layer.post_mlp_bda
    
    def _init_layer_tensor_parallel_comm(self):
        comm_tensor1 = self.attention_layers[0].get_persistent_outputs_bwd(1) # TODO: first layer
        num_layers = len(self.attention_layers)
        for l_no in range(num_layers):
            attention_layer = self.attention_layers[l_no]
            mlp_layer = self.mlp_layers[l_no]

            attention_layer.init_tensor_parallel_comm_fwd(1, comm_tensor1)
            
            current_hidden_1 = attention_layer.get_persistent_outputs_fwd(1)
            comm_tensor2 = current_hidden_1
            attention_layer.init_tensor_parallel_comm_fwd(2, comm_tensor2)

            current_hidden_2 = attention_layer.get_persistent_outputs_fwd(2)
            comm_tensor1 = current_hidden_2
            mlp_layer.init_tensor_parallel_comm_fwd(1, comm_tensor1)

            current_hidden_1 = mlp_layer.get_persistent_outputs_fwd(1)
            comm_tensor2 = current_hidden_1
            mlp_layer.init_tensor_parallel_comm_fwd(2, comm_tensor2)

            current_hidden_2 = mlp_layer.get_persistent_outputs_fwd(2)
            comm_tensor1 = current_hidden_2
        
        comm_tensor2 = self.mlp_layers[-1].get_persistent_outputs_fwd(2) # TODO: last layer
        for l_no in range(num_layers - 1, -1, -1):
            mlp_layer = self.mlp_layers[l_no]
            attention_layer = self.attention_layers[l_no]
            
            mlp_layer.init_tensor_parallel_comm_bwd(2, comm_tensor2)

            current_grad_2 = mlp_layer.get_persistent_outputs_bwd(2)
            comm_tensor1 = current_grad_2
            mlp_layer.init_tensor_parallel_comm_bwd(1, comm_tensor1)

            current_grad_1 = mlp_layer.get_persistent_outputs_bwd(1)
            comm_tensor2 = current_grad_1
            attention_layer.init_tensor_parallel_comm_bwd(2, comm_tensor2)

            current_grad_2 = attention_layer.get_persistent_outputs_bwd(2)
            comm_tensor1 = current_grad_2
            attention_layer.init_tensor_parallel_comm_bwd(1, comm_tensor1)

            current_grad_1 = attention_layer.get_persistent_outputs_bwd(1)
            comm_tensor2 = current_grad_1
        
        for l_no in range(num_layers):
            attention_layer = self.attention_layers[l_no]
            mlp_layer = self.mlp_layers[l_no]

            attention_layer.build_attention_fuser()
            mlp_layer.build_mlp_fuser()
     

    def _get_attention_layer(self, layer_number: int):
        return self.attention_layers[layer_number]
    
    def _get_mlp_layer(self, layer_number: int):
        return self.mlp_layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def _split_tensors_for_nanobatch(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None, # TODO: mid_point split is wrong
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> tuple:
        """
        Split input tensors into two halves for nano-batch processing.
        
        Returns:
            tuple: Two tuples containing the split tensors for each half
        """
        batch_size = hidden_states.size(1)
        if batch_size < 2:
            raise ValueError(f"Batch size must be at least 2 for nano-batch splitting, got {batch_size}")
        
        mid_point = batch_size // 2

        assert attention_mask is None
        assert context is None
        assert context_mask is None
        assert sequence_len_offset is None
        assert attention_bias is None
        assert sequence_len_offset is None
        
        # Split input tensors
        hidden_states_1 = hidden_states[:, :mid_point, ...]
        hidden_states_2 = hidden_states[:, mid_point:, ...]

        attention_mask_1 = None
        attention_mask_2 = None
        context_1 = None
        context_2 = None
        context_mask_1 = None
        context_mask_2 = None
        attention_bias_1 = None
        attention_bias_2 = None
        sequence_len_offset_1 = None
        sequence_len_offset_2 = None
        
        # attention_mask_1 = attention_mask[:, :, :, :mid_point] if attention_mask is not None else None
        # attention_mask_2 = attention_mask[:, :, :, mid_point:] if attention_mask is not None else None
        
        # context_1 = context[:, :mid_point, ...] if context is not None else None
        # context_2 = context[:, mid_point:, ...] if context is not None else None
        
        # context_mask_1 = context_mask[:, :mid_point, ...] if context_mask is not None else None
        # context_mask_2 = context_mask[:, mid_point:, ...] if context_mask is not None else None
        
        # attention_bias_1 = attention_bias[:mid_point, ...] if attention_bias is not None else None
        # attention_bias_2 = attention_bias[mid_point:, ...] if attention_bias is not None else None
        
        # sequence_len_offset_1 = sequence_len_offset[:mid_point] if sequence_len_offset is not None else None
        # sequence_len_offset_2 = sequence_len_offset[mid_point:] if sequence_len_offset is not None else None
        
        return (
            (hidden_states_1, attention_mask_1, context_1, context_mask_1, attention_bias_1, sequence_len_offset_1),
            (hidden_states_2, attention_mask_2, context_2, context_mask_2, attention_bias_2, sequence_len_offset_2)
        )

    def _checkpointed_forward(
        self,
        *,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor],
        attention_bias: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> Tensor:
        """
        Activation-checkpointed forward path. Only supports recompute_granularity == 'full'.

        We checkpoint the entire block execution in a single function to avoid
        per-layer state plumbing for the custom nano-batch interleaving.
        """
        

        def run_layers(
            h1: Tensor,
            h2: Tensor,
            am1: Optional[Tensor],
            am2: Optional[Tensor],
            c1: Optional[Tensor],
            c2: Optional[Tensor],
            cm1: Optional[Tensor],
            cm2: Optional[Tensor],
            ab1: Optional[Tensor],
            ab2: Optional[Tensor],
            psp: Optional[PackedSeqParams],
        ):
            # Local copies of closures that may be None
            current_hidden_1 = h1
            current_hidden_2 = h2
            current_context_1 = c1
            current_context_2 = c2

            # Initialize residual/comm as in the non-checkpoint forward
            residual_1 = h2
            residual_2 = h1
            comm_hidden_1 = (h2, None)

            for l_no in range(len(self.attention_layers)):
                attention_layer = self.attention_layers[l_no]
                mlp_layer = self.mlp_layers[l_no]

                # Attention pass - micro-batch 1
                current_hidden_1, residual_1, comm_hidden_1, current_context_1 = attention_layer(
                    batch_idx=1,
                    hidden_states=current_hidden_1,
                    residual=residual_1,
                    comm_hidden_states=comm_hidden_1,
                    attention_mask=am1,
                    context=current_context_1,
                    context_mask=cm1,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=ab1,
                    inference_context=inference_context,
                    packed_seq_params=psp,
                )
                comm_hidden_2 = current_hidden_1
                current_hidden_2 = comm_hidden_1 if not l_no == 0 else current_hidden_2

                # Attention pass - micro-batch 2
                current_hidden_2, residual_2, comm_hidden_2, current_context_2 = attention_layer(
                    batch_idx=2,
                    hidden_states=current_hidden_2,
                    residual=residual_2,
                    comm_hidden_states=comm_hidden_2,
                    attention_mask=am2,
                    context=current_context_2,
                    context_mask=cm2,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=ab2,
                    inference_context=inference_context,
                    packed_seq_params=psp,
                )
                comm_hidden_1 = current_hidden_2
                current_hidden_1 = comm_hidden_2

                # MLP pass - micro-batch 1
                current_hidden_1, residual_1, comm_hidden_1 = mlp_layer(
                    batch_idx=1,
                    hidden_states=current_hidden_1,
                    residual=residual_1,
                    comm_hidden_states=comm_hidden_1,
                )
                comm_hidden_2 = current_hidden_1
                current_hidden_2 = comm_hidden_1

                # MLP pass - micro-batch 2
                current_hidden_2, residual_2, comm_hidden_2 = mlp_layer(
                    batch_idx=2,
                    hidden_states=current_hidden_2,
                    residual=residual_2,
                    comm_hidden_states=comm_hidden_2,
                )
                comm_hidden_1 = current_hidden_2
                current_hidden_1 = comm_hidden_2

            return current_hidden_1, current_hidden_2, current_context_1, current_context_2

        def checkpoint_handler(forward_func):
            # Match the original calling convention: pass a single set of canonical inputs
            # and keep nano-batch splits in the closure.
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    attention_bias,
                    packed_seq_params,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    attention_bias,
                    packed_seq_params,
                )

        # Accept canonical inputs and recompute nano-batch splits inside the checkpointed func
        def custom_forward(_hs, _am, _ctx, _cm, _rpe, _ab, _psp):
            (
                hs1, am1, ctx1, cm1, ab1, _,
            ), (
                hs2, am2, ctx2, cm2, ab2, _,
            ) = self._split_tensors_for_nanobatch(
                _hs, _am, _ctx, _cm, _ab, None
            )
            out_h1, out_h2, _, _ = run_layers(hs1, hs2, am1, am2, ctx1, ctx2, cm1, cm2, ab1, ab2, _psp)
            hidden_states_1 = out_h1[0]
            hidden_states_2 = out_h2[0]
            return torch.cat([hidden_states_1, hidden_states_2], dim=1)

        hidden_states = checkpoint_handler(custom_forward)
        return hidden_states

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
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Update the inference parameters with the current batch size in case it is variable
        if inference_context and not self.training:
            inference_context.current_batch_size = hidden_states.size(1)

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        # Process nano-batches with interleaved execution:
        # 1. Execute attention for batch 1
        # 2. Execute attention for batch 2  
        # 3. Execute MLP for batch 1
        # 4. Execute MLP for batch 2
        
        if self.config.sequence_parallel:
            raise NotImplementedError("Sequence parallel not implemented")
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with rng_context, outer_fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                )
            else:
                (hidden_states_1, attention_mask_1, context_1, context_mask_1, attention_bias_1, sequence_len_offset_1), \
                (hidden_states_2, attention_mask_2, context_2, context_mask_2, attention_bias_2, sequence_len_offset_2) = \
                    self._split_tensors_for_nanobatch(
                        hidden_states, attention_mask, context, context_mask, attention_bias, sequence_len_offset
                    )
                # Layer-by-layer execution: For each layer, process both nano-batches
                # through attention, then both nano-batches through MLP
                current_hidden_1 = hidden_states_1
                current_hidden_2 = hidden_states_2
                residual_1 = hidden_states_2  # TODO: for first layer
                residual_2 = hidden_states_1
                comm_hidden_1 = (hidden_states_2, None) # TODO: first layer
                current_context_1 = context_1
                current_context_2 = context_2
                
                for l_no in range(len(self.attention_layers)):
                    attention_layer = self.attention_layers[l_no]
                    mlp_layer = self.mlp_layers[l_no]
                    
                    inner_fp8_context = (
                        get_fp8_context(self.config, attention_layer.layer_number - 1)
                        if use_inner_fp8_context
                        else nullcontext()
                    )
                    
                    with self.offload_context, inner_fp8_context:
                        # Process attention for both nano-batches
                        # Micro-batch 1 attention
                        current_hidden_1, residual_1, comm_hidden_1, current_context_1 = attention_layer(
                            batch_idx=1,
                            hidden_states=current_hidden_1,
                            residual=residual_1,
                            comm_hidden_states=comm_hidden_1,
                            attention_mask=attention_mask_1,
                            context=current_context_1,
                            context_mask=context_mask_1,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias_1,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset_1,
                        )
                        comm_hidden_2 = current_hidden_1
                        # current_hidden_2 = comm_hidden_1 if comm_hidden_1 is not None else current_hidden_2
                        current_hidden_2 = comm_hidden_1 if not l_no == 0 else current_hidden_2
                        
                        # Micro-batch 2 attention
                        current_hidden_2, residual_2, comm_hidden_2, current_context_2 = attention_layer(
                            batch_idx=2,
                            hidden_states=current_hidden_2,
                            residual=residual_2,
                            comm_hidden_states=comm_hidden_2,
                            attention_mask=attention_mask_2,
                            context=current_context_2,
                            context_mask=context_mask_2,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias_2,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset_2,
                        )
                        comm_hidden_1 = current_hidden_2
                        current_hidden_1 = comm_hidden_2
                        
                        # Process MLP for both nano-batches
                        # Micro-batch 1 MLP
                        current_hidden_1, residual_1, comm_hidden_1 = mlp_layer(
                            batch_idx=1,
                            hidden_states=current_hidden_1,
                            residual=residual_1,
                            comm_hidden_states=comm_hidden_1
                        )
                        comm_hidden_2 = current_hidden_1
                        current_hidden_2 = comm_hidden_1
                        
                        # Micro-batch 2 MLP
                        current_hidden_2, residual_2, comm_hidden_2 = mlp_layer(
                            batch_idx=2,
                            hidden_states=current_hidden_2,
                            residual=residual_2,
                            comm_hidden_states=comm_hidden_2
                        )
                        comm_hidden_1 = current_hidden_2
                        current_hidden_1 = comm_hidden_2
                        
                        if (
                            torch.is_grad_enabled()
                            and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None
                        ):
                            raise NotImplementedError("CPU offloading not implemented")
                
                hidden_states_1 = current_hidden_1[0]
                hidden_states_2 = current_hidden_2[0]
                # Concatenate results
                hidden_states = torch.cat([hidden_states_1, hidden_states_2], dim=1)
                # TODO: final allreduce and BDA
                
                # # Set context from the last processed context
                # if current_context_1 is not None and current_context_2 is not None:
                #     context = torch.cat([current_context_1, current_context_2], dim=1)
                # elif current_context_1 is not None:
                #     context = current_context_1
                # elif current_context_2 is not None:
                #     context = current_context_2
                # else:
                #     context = None

        # Final layer norm for both nano-batches
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
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

        # Handle attention layers
        attention_layer_prefix = f'{prefix}attention_layers.'
        num_layers = self.config.num_layers
        for i, attention_layer in enumerate(self.attention_layers):
            offset = get_transformer_layer_offset(self.config)

            global_layer_offset = attention_layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{attention_layer_prefix}{i}.'  # module list index in TransformerBlock
            if non_homogeneous_layers:
                sharded_prefix = f'{attention_layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = attention_layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = attention_layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Handle MLP layers
        mlp_layer_prefix = f'{prefix}mlp_layers.'
        for i, mlp_layer in enumerate(self.mlp_layers):
            offset = get_transformer_layer_offset(self.config)

            global_layer_offset = mlp_layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{mlp_layer_prefix}{i}.'  # module list index in TransformerBlock
            if non_homogeneous_layers:
                sharded_prefix = f'{mlp_layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = mlp_layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = mlp_layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.attention_layers and self.mlp_layers
        for name, module in self.named_children():
            if not (module is self.attention_layers or module is self.mlp_layers):
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict

    @property
    def layers(self):
        """
        Backward compatibility property for pipeline parallel scheduler.
        Returns a combined view of attention and MLP layers.
        """
        # Create a combined list that alternates attention and MLP layers
        # This maintains the expected interface for pipeline parallel code
        combined_layers = []
        for i in range(len(self.attention_layers)):
            # Add a wrapper that combines attention and MLP for this layer
            combined_layer = CombinedLayerWrapper(self.attention_layers[i], self.mlp_layers[i])
            combined_layers.append(combined_layer)
        return combined_layers
