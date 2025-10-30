from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    get_transformer_layer_offset,
    BaseTransformerLayer,
)
from megatron.core.utils import deprecate_inference_params, make_viewless_tensor
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from kareus.utils.debug import save_tensors
from kareus.transformer_engine.pytorch.ops import AllReduce
from kareus.megatron.core.extensions.partition_fuser import PartitionFuser
from kareus.megatron.core.extensions.attn_oproj_fuser import AttnOprojPartitionFuser
from kareus.megatron.core.extensions.qkv_fuser import QKVPartitionFuser
from kareus.megatron.core.extensions.qkv_fuser2 import QKVPartitionFuser2

@dataclass
class AttentionLayerSubmodules:
    """Configuration class for specifying the submodules of an attention layer."""
    
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    post_self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    # pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    # cross_attention: Union[ModuleSpec, type] = IdentityOp
    # cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


def get_fuser_comm_kwargs(config: TransformerConfig):
    comm_scheduler = config.kareus_scheduler
    if comm_scheduler is None:
        return {
            "comm_overlap_window": (0, 8),
            "comm_sm_configs": (None, None),
            "comm_overlap_window_backward": (0, 8),
            "comm_sm_configs_backward": (None, None),
        }
    else:
        item = getattr(comm_scheduler, "current_schedule", None)
        if item is None:
            print("current_schedule is not set")
            return {
                "comm_overlap_window": (2, 8),
                "comm_sm_configs": (6, 1024),
                "comm_overlap_window_backward": (0, 8),
                "comm_sm_configs_backward": (6, 1024),
            }
        fwd_attn = item.fwd_attn
        bwd_attn = item.bwd_attn
        return {
            "comm_overlap_window": None if fwd_attn is None else fwd_attn.overlap_window,
            "comm_sm_configs": None if fwd_attn is None else fwd_attn.resource_shape,
            "comm_overlap_window_backward": None if bwd_attn is None else bwd_attn.overlap_window,
            "comm_sm_configs_backward": None if bwd_attn is None else bwd_attn.resource_shape,
        }


def get_fuser_comm_kwargs_cp(config: TransformerConfig, fuser_type: str, is_first_layer: bool = False):
    comm_scheduler = config.kareus_scheduler
    if is_first_layer:
        if fuser_type == "qkv_ag":
            return {
                "comm_overlap_window": (-1, -1),
                "comm_sm_configs": (12, 1024),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (12, 1024),
            }
        if fuser_type == "ao_ag":
            return {
                "comm_overlap_window": (-1, -1),
                "comm_sm_configs": (12, 1024),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (12, 1024),
            }
        # if fuser_type == "ao_ar":
        #     return {
        #         "comm_overlap_window": (-1, -1),
        #         "comm_sm_configs": (None, None),
        #         "comm_overlap_window_backward": (0, -1),
        #         "comm_sm_configs_backward": (12, 1024),
        #     }

    if comm_scheduler is None:
        if fuser_type == "qkv_ar":
            return {
                "comm_overlap_window": (0, -1),  # comm_end doesn't matter
                "comm_sm_configs": (12, 1024),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (12, 1024),
            }
        elif fuser_type == "qkv_ag":
            return {
                "comm_overlap_window": (0, -1),
                "comm_sm_configs": (12, 1024),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (12, 1024),
            }
        else:
            return {
                "comm_overlap_window_ao_ag": (0, -1),
                "comm_sm_configs_ao_ag": (12, 1024),
                "comm_overlap_window_ao_ar": (0, -1),
                "comm_sm_configs_ao_ar": (12, 1024),
                "comm_overlap_window_a_rs": (0, -1),
                "comm_sm_configs_a_rs": (12, 1024),
                "comm_overlap_window_a_ag": (0, -1),
                "comm_sm_configs_a_ag": (12, 1024),
                "comm_overlap_window_o_ag": (0, -1),
                "comm_sm_configs_o_ag": (12, 1024),
                "comm_overlap_window_o_ar": (0, -1),
                "comm_sm_configs_o_ar": (12, 1024),
            }
    else:
        item = getattr(comm_scheduler, "current_schedule", None)
        if item is None:
            raise ValueError("current_schedule is not set")

        fwd_qkv_ar = item.fwd_qkv_ar
        fwd_qkv_ag = item.fwd_qkv_ag
        fwd_ao_ag = item.fwd_ao_ag
        fwd_ao_ar = item.fwd_ao_ar
        bwd_qkv_ar = item.bwd_qkv_ar
        bwd_qkv_rs = item.bwd_qkv_rs
        bwd_a_rs = item.bwd_a_rs
        bwd_a_ag = item.bwd_a_ag
        bwd_o_ag = item.bwd_o_ag
        bwd_o_ar = item.bwd_o_ar
        if fuser_type == "qkv_ar":
            return {
                "comm_overlap_window": None if fwd_qkv_ar is None else fwd_qkv_ar.overlap_window,
                "comm_sm_configs": None if fwd_qkv_ar is None else fwd_qkv_ar.resource_shape,
                "comm_overlap_window_backward": None if bwd_qkv_ar is None else bwd_qkv_ar.overlap_window,
                "comm_sm_configs_backward": None if bwd_qkv_ar is None else bwd_qkv_ar.resource_shape,
            }
        elif fuser_type == "qkv_ag":
            return {
                "comm_overlap_window": None if fwd_qkv_ag is None else fwd_qkv_ag.overlap_window,
                "comm_sm_configs": None if fwd_qkv_ag is None else fwd_qkv_ag.resource_shape,
                "comm_overlap_window_backward": None if bwd_qkv_rs is None else bwd_qkv_rs.overlap_window,
                "comm_sm_configs_backward": None if bwd_qkv_rs is None else bwd_qkv_rs.resource_shape,
            }
        else:
            return {
                "comm_overlap_window_ao_ag": None if fwd_ao_ag is None else fwd_ao_ag.overlap_window,
                "comm_sm_configs_ao_ag": None if fwd_ao_ag is None else fwd_ao_ag.resource_shape,
                "comm_overlap_window_ao_ar": None if fwd_ao_ar is None else fwd_ao_ar.overlap_window,
                "comm_sm_configs_ao_ar": None if fwd_ao_ar is None else fwd_ao_ar.resource_shape,
                "comm_overlap_window_a_rs": None if bwd_a_rs is None else bwd_a_rs.overlap_window,
                "comm_sm_configs_a_rs": None if bwd_a_rs is None else bwd_a_rs.resource_shape,
                "comm_overlap_window_a_ag": None if bwd_a_ag is None else bwd_a_ag.overlap_window,
                "comm_sm_configs_a_ag": None if bwd_a_ag is None else bwd_a_ag.resource_shape,
                "comm_overlap_window_o_ag": None if bwd_o_ag is None else bwd_o_ag.overlap_window,
                "comm_sm_configs_o_ag": None if bwd_o_ag is None else bwd_o_ag.resource_shape,
                "comm_overlap_window_o_ar": None if bwd_o_ar is None else bwd_o_ar.overlap_window,
                "comm_sm_configs_o_ar": None if bwd_o_ar is None else bwd_o_ar.resource_shape,
            }


class AttentionLayer(MegatronModule, BaseTransformerLayer):
    """Attention layer containing self-attention operations."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AttentionLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph or config.external_cuda_graph:
            raise NotImplementedError("Cuda graph not implemented")

        self.config = config
        self.submodules_config = submodules
        self.is_first_layer = layer_number == 1
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # self.tp_comms = [] # [[fwd_comm, bwd_comm], [fwd_comm, bwd_comm]]
        self.tp_comms = [] # [allreduce, allreduce] for two nano-batches
        self.cp_comms = [] # [[allgather, allgather], [reducescatter, reducescatter]]
        self.attention_fusers = []

        # [Module 1: Prev MLP BDA] Optional BDA on the previous MLP output
        self.prev_mlp_bda = None

        # [Module 2: Input Layernorm] Optional Layernorm on the input data
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # self.input_layernorm = create_operation_fuser(self.input_layernorm)

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        
        # [Module 3: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 4: Post Self-Attention BDA] Optional BDA on the Self-Attention output
        self.post_self_attn_bda = build_module(submodules.post_self_attn_bda)
    
        self.recompute_input_layernorm = False
        if self.config.recompute_granularity == 'selective':
            raise NotImplementedError("Selective recompute not implemented")
        
        # Set bias+dropout+add fusion grad_enable execution handler.
        # Note: BiasDropoutAddOp now handles torch.enable_grad() internally
        # self.bias_dropout_add_exec_handler = torch.enable_grad

    # def build_attention_fuser(self, batch_idx):  # TODO: handle different layers
    #     # if self.is_first_layer:
    #     #     comp_ops = [self.input_layernorm]
    #     # else:
    #     #     comp_ops = [self.post_self_attn_bda, self.input_layernorm]
    #     comp_ops = [self.post_self_attn_bda, self.input_layernorm] # TODO: first layer
    #     comp_ops.extend(self.self_attention.get_compute_ops())
    #     attention_fuser = PartitionFuser(
    #         ops=comp_ops,
    #         comm_op_fwd=self.tp_comms[batch_idx - 1][0],
    #         comm_op_bwd=self.tp_comms[batch_idx - 1][1],
    #         fuse_ops=False,
    #     )
    #     return attention_fuser

    def build_attention_fuser(self):
        assert len(self.tp_comms) == 2, "tp_comms is not initialized"
        # if self.is_first_layer:
        #     comp_ops = [self.input_layernorm]
        # else:
        #     comp_ops = [self.post_self_attn_bda, self.input_layernorm]
        comp_ops = [self.post_self_attn_bda, self.input_layernorm] # TODO: first layer
        comp_ops.extend(self.self_attention.get_compute_ops())
        context_parallel = self.config.context_parallel_size > 1
        if not context_parallel:
            for i in range(len(self.tp_comms)):
                # self.attention_fusers.append(
                #     PartitionFuser(
                #         ops=comp_ops,
                #         comm_op_fwd=self.tp_comms[i][0],
                #         comm_op_bwd=self.tp_comms[i][1],
                #         fuse_ops=False,
                #         is_first_attn=self.is_first_layer and i == 0,
                #     )
                # )
                fwd_comm = self.tp_comms[i]
                bwd_comm = self.tp_comms[i]
                if self.is_first_layer and i == 0:
                    fwd_comm = None
                self.attention_fusers.append(
                    PartitionFuser(
                        ops=comp_ops,
                        comm_op_fwd=fwd_comm,
                        comm_op_bwd=bwd_comm,
                        fuse_ops=False,
                        is_first_attn=self.is_first_layer and i == 0,
                    )
                )
        else:
            if len(self.cp_comms) == 0:
                return
            qkv_comp_ops = comp_ops[:5]
            qkv_ar_fuser = QKVPartitionFuser(
                ops=qkv_comp_ops,
                comm_op_fwd=self.tp_comms[0] if not self.is_first_layer else None,
                comm_op_bwd=self.tp_comms[0],
                fuse_ops=False,
                is_first_attn=self.is_first_layer,
            )
            qkv_ag_fuser = QKVPartitionFuser2(
                ops=qkv_comp_ops,
                comm_op_fwd=self.cp_comms[0][0],
                comm_op_bwd=self.cp_comms[1][0],
                fuse_ops=False,
            )
            ao_comp_ops = comp_ops[5:]
            comm_op_fwd = [self.cp_comms[0][1], self.tp_comms[1]]
            comm_op_bwd = [self.cp_comms[1][1], self.cp_comms[0][0], self.cp_comms[0][1], self.tp_comms[1]]
            ao_fuser = AttnOprojPartitionFuser(
                ops=ao_comp_ops,
                comm_ops_fwd=comm_op_fwd,
                comm_ops_bwd=comm_op_bwd,
                fuse_ops=False,
            )
            self.attention_fusers = [qkv_ar_fuser, qkv_ag_fuser, ao_fuser]
    
    def init_tensor_parallel_comm_fwd(self, batch_idx, comm_tensor):
        if self.is_first_layer and batch_idx == 1: # TODO: first layer
            fwd_comm = None
        else:
            fwd_comm = AllReduce(
                process_group=get_tensor_model_parallel_group(check_initialized=False),
                async_op=True,
                backend="msccl",
                rank=get_tensor_model_parallel_rank(),
                world_size=get_tensor_model_parallel_world_size(),
                use_persistent_output=True,
                input_buffer=comm_tensor,
            )
        assert len(self.tp_comms) == batch_idx - 1, "batch_idx is not correct"
        self.tp_comms.append([fwd_comm, None])
    
    def init_tensor_parallel_comm_bwd(self, batch_idx, comm_tensor):
        bwd_comm = AllReduce(
            process_group=get_tensor_model_parallel_group(check_initialized=False),
            async_op=True,
            backend="msccl",
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_world_size(),
            use_persistent_output=True,
            input_buffer=comm_tensor,
        )
        self.tp_comms[batch_idx - 1][1] = bwd_comm
    
    def init_context_parallel_comm(self, allgather_comm_ops, reducescatter_comm_ops):
        self.cp_comms = [allgather_comm_ops, reducescatter_comm_ops]

    def init_tensor_parallel_comm(self, allreduce_comm_ops):
        self.tp_comms = allreduce_comm_ops

    def get_persistent_outputs_fwd(self, batch_idx: int):
        return self.self_attention.get_persistent_outputs_fwd()[batch_idx - 1]
    
    def get_persistent_outputs_bwd(self, batch_idx: int):
        return self.self_attention.get_persistent_outputs_bwd()[batch_idx - 1]
        
    def forward(
        self,
        batch_idx: int,
        hidden_states: Union[Tensor, Tuple[Tensor, Tensor]],
        residual: Tensor = None,
        comm_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used.
        """
        assert context is None, "context is not supported"
        # attention_fuser = self.build_attention_fuser(batch_idx)

        context_parallel = self.config.context_parallel_size > 1
        if not context_parallel:

            if self.is_first_layer:
                hidden_states = hidden_states
                bias = None
                # residual = hidden_states
            else:
                hidden_states, bias = hidden_states

            output, output_bias, output_residual, allreduce_output = self.attention_fusers[batch_idx - 1](  # TODO: add dropout_prob and training
                hidden_states=hidden_states,
                bias=bias,
                residual=residual,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                comm_input=comm_hidden_states[0] if comm_hidden_states is not None else None,
                **get_fuser_comm_kwargs(self.config),
            )
            output_hidden_states = (output, output_bias)
            allreduce_output = (allreduce_output, comm_hidden_states[1]) if comm_hidden_states is not None else None
            # allreduce_output = (allreduce_output, comm_hidden_states[1])
            return output_hidden_states, output_residual, allreduce_output, context
        
        else:

            if self.is_first_layer:
                hidden_states_1, hidden_states_2 = hidden_states
                bias_1, bias_2 = None, None
                comm_input_1 = hidden_states_2
            else:
                hidden_states_1, bias_1 = hidden_states
                comm_input_1, bias_2 = comm_hidden_states
            residual_1, residual_2 = residual

            qkv_ar_fuser, qkv_ag_fuser, ao_fuser = self.attention_fusers

            query_1, key_1, value_1, residual_1, allreduce_output_1 = qkv_ar_fuser(
                hidden_states=hidden_states_1,
                bias=bias_1,
                residual=residual_1,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                comm_input=comm_input_1,
                **get_fuser_comm_kwargs_cp(self.config, "qkv_ar", self.is_first_layer)
            )
            hidden_states_2 = allreduce_output_1
            
            query_2, key_2, value_2, residual_2 = qkv_ag_fuser(
                hidden_states=hidden_states_2,
                bias=bias_2,
                residual=residual_2,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                comm_key=key_1,
                comm_value=value_1,
                **get_fuser_comm_kwargs_cp(self.config, "qkv_ag", self.is_first_layer)
            )

            out_1, out_2, bias_1, bias_2 = ao_fuser(
                query_1=query_1,
                query_2=query_2,
                comm_key=key_2,
                comm_value=value_2,
                **get_fuser_comm_kwargs_cp(self.config, "ao", self.is_first_layer)
            )

            return (out_1, bias_1), (out_2, bias_2), residual_1, residual_2

            
        # inference_context = deprecate_inference_params(inference_context, inference_params)

        # # if comm_hidden_states is not None:
        # #     comm_tensor, bias = comm_hidden_states
        # #     comm_tensor = self.tp_comm(comm_tensor)
        # #     self.tp_comm.sync()
        # #     comm_hidden_states = (comm_tensor, bias)

        # if not self.is_first_layer:
        #     assert self.prev_mlp_bda is not None, "prev_mlp_bda is not initialized"
        #     hidden_states = self.prev_mlp_bda(hidden_states[0], hidden_states[1], residual,
        #                                     training=self.training, dropout_prob=self.hidden_dropout)

        # # Residual connection.
        # residual = hidden_states

        # # Optional Input Layer norm
        # input_layernorm_output = self.input_layernorm(hidden_states)

        # # Self attention.
        # attention_output_with_bias = self.self_attention(
        #     batch_idx - 1,
        #     input_layernorm_output,
        #     attention_mask=attention_mask,
        #     inference_context=inference_context,
        #     rotary_pos_emb=rotary_pos_emb,
        #     rotary_pos_cos=rotary_pos_cos,
        #     rotary_pos_sin=rotary_pos_sin,
        #     attention_bias=attention_bias,
        #     packed_seq_params=packed_seq_params,
        #     sequence_len_offset=sequence_len_offset,
        # )

        # return attention_output_with_bias, residual, comm_hidden_states, context
    
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the attention layer."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
    