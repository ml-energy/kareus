
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

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
    TransformerLayerSubmodules,
)
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.utils import make_viewless_tensor

from kareus.megatron.core.extensions.fusers.partition_fuser import PartitionFuser
from kareus.megatron.core.extensions.fusers.attn_oproj_fuser import AttnOprojPartitionFuser
from kareus.megatron.core.extensions.fusers.qkv_fuser import QKVPartitionFuser
from kareus.megatron.core.extensions.fusers.qkv_fuser2 import QKVPartitionFuser2
from kareus.megatron.core.extensions.ops.residual_fork import ResidualForkOp


# ---------------------------------------------------------------------------
# Fuser communication kwargs helpers
# ---------------------------------------------------------------------------

def _get_attn_fuser_comm_kwargs(config: TransformerConfig):
    """Get communication kwargs for attention fusers (TP-only mode)."""
    comm_scheduler = config.kareus_scheduler
    if comm_scheduler is None:
        return {
            "comm_overlap_window": (0, 8),
            "comm_sm_configs": (None, None),
            "comm_overlap_window_backward": (0, 8),
            "comm_sm_configs_backward": (None, None),
        }

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


def _get_attn_fuser_comm_kwargs_cp(
    config: TransformerConfig,
    fuser_type: str,
    is_first_layer: bool = False,
):
    """Get communication kwargs for attention fusers (CP mode)."""
    comm_scheduler = config.kareus_scheduler

    if comm_scheduler is None:
        if fuser_type == "qkv_ar":
            return {
                "comm_overlap_window": (0, -1),
                "comm_sm_configs": (None, None),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (None, None),
            }
        elif fuser_type == "qkv_ag":
            return {
                "comm_overlap_window": (0, -1),
                "comm_sm_configs": (None, None),
                "comm_overlap_window_backward": (0, -1),
                "comm_sm_configs_backward": (None, None),
            }
        else:
            return {
                "comm_overlap_window_ao_ag": (0, -1),
                "comm_sm_configs_ao_ag": (None, None),
                "comm_overlap_window_ao_ar": (0, -1),
                "comm_sm_configs_ao_ar": (None, None),
                "comm_overlap_window_a_rs": (0, -1),
                "comm_sm_configs_a_rs": (None, None),
                "comm_overlap_window_a_ag": (0, -1),
                "comm_sm_configs_a_ag": (None, None),
                "comm_overlap_window_o_ag": (0, -1),
                "comm_sm_configs_o_ag": (None, None),
                "comm_overlap_window_o_ar": (0, -1),
                "comm_sm_configs_o_ar": (None, None),
            }

    item = getattr(comm_scheduler, "current_schedule", None)
    if item is None:
        if fuser_type == "qkv_ar":
            return {
                "comm_overlap_window": (0, -1),
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


def _get_mlp_fuser_comm_kwargs(config: TransformerConfig):
    """Get communication kwargs for MLP fusers."""
    comm_scheduler = config.kareus_scheduler
    if comm_scheduler is None:
        return {
            "comm_overlap_window": (0, 6),
            "comm_sm_configs": (None, None),
            "comm_overlap_window_backward": (0, 6),
            "comm_sm_configs_backward": (None, None),
        }

    item = getattr(comm_scheduler, "current_schedule", None)
    if item is None:
        return {
            "comm_overlap_window": (2, 6),
            "comm_sm_configs": (6, 1024),
            "comm_overlap_window_backward": (0, 6),
            "comm_sm_configs_backward": (6, 1024),
        }

    fwd_mlp = item.fwd_mlp
    bwd_mlp = item.bwd_mlp
    return {
        "comm_overlap_window": None if fwd_mlp is None else fwd_mlp.overlap_window,
        "comm_sm_configs": None if fwd_mlp is None else fwd_mlp.resource_shape,
        "comm_overlap_window_backward": None if bwd_mlp is None else bwd_mlp.overlap_window,
        "comm_sm_configs_backward": None if bwd_mlp is None else bwd_mlp.resource_shape,
    }


# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer with Kareus partition-aware execution.

    Follows the structure of Megatron-LM's TransformerLayer (containing both
    self-attention and MLP submodules) but uses Kareus's partition fusers for
    communication-overlapped execution with nano-batch interleaving.

    Submodules (same as Megatron's TransformerLayerSubmodules):
        input_layernorm  -> self_attention -> self_attn_bda
        pre_mlp_layernorm -> mlp           -> mlp_bda

    Execution model:
        TransformerBlock calls ``forward_attention`` and ``forward_mlp``
        separately for each nano-batch, enabling the interleaved
        compute/communication overlap pattern.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph or config.external_cuda_graph:
            raise NotImplementedError(
                "CUDA graph not implemented for Kareus TransformerLayer"
            )

        if (
            submodules.pre_cross_attn_layernorm is not IdentityOp
            or submodules.cross_attention is not IdentityOp
            or submodules.cross_attn_bda is not IdentityFuncOp
        ):
            raise NotImplementedError(
                "Cross attention is not supported in Kareus TransformerLayer"
            )

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = (
            config.hidden_dropout if hidden_dropout is None else hidden_dropout
        )

        num_layers = get_num_layers_to_build(config)
        self.is_first_layer = layer_number == 1
        self.is_last_layer = layer_number == num_layers

        # Communication operators (populated by init_*_comm methods)
        self.tp_comms: List = []   # [allreduce_nb0, allreduce_nb1]
        self.cp_comms: List = []   # [[ag_nb0, ag_nb1], [rs_nb0, rs_nb1]]

        # Fusers (populated by build_fusers)
        self.attention_fusers: List = []
        self.mlp_fusers: List = []

        # =================================================================
        # Attention submodules
        # =================================================================

        self.attn_residual_fork = ResidualForkOp()

        # [Module 1: Input Layernorm]
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        # [Module 2: Self Attention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: Self-Attention BiasDropoutAdd]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # =================================================================
        # MLP submodules
        # =================================================================

        self.mlp_residual_fork = ResidualForkOp()

        # [Module 4: Pre-MLP Layernorm]
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: MLP]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 6: MLP BiasDropoutAdd]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # Recompute flags
        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == "selective":
            raise NotImplementedError(
                "Selective recompute not implemented for Kareus TransformerLayer"
            )

    # =================================================================
    # Communication initialization
    # =================================================================

    def init_tensor_parallel_comm(self, allreduce_comm_ops: List) -> None:
        """Set shared AllReduce operators (one per nano-batch)."""
        self.tp_comms = allreduce_comm_ops

    def init_context_parallel_comm(
        self, allgather_comm_ops: List, reducescatter_comm_ops: List
    ) -> None:
        """Set shared AllGather/ReduceScatter operators for context parallelism."""
        self.cp_comms = [allgather_comm_ops, reducescatter_comm_ops]

    # =================================================================
    # Fuser building
    # =================================================================

    def build_fusers(self) -> None:
        """Build all partition fusers for this layer (attention + MLP)."""
        self._build_attention_fusers()
        self._build_mlp_fusers()

    def _build_attention_fusers(self) -> None:
        """Build attention partition fusers.

        TP-only mode: creates one PartitionFuser per nano-batch.
        CP mode: creates [QKVPartitionFuser, QKVPartitionFuser2,
                          AttnOprojPartitionFuser].
        """
        assert len(self.tp_comms) == 2, "tp_comms not initialised"

        # Attention partition ops: BDA -> LayerNorm -> QKV -> ... -> OProj
        comp_ops = [self.self_attn_bda, self.input_layernorm]
        comp_ops.extend(self.self_attention.get_compute_ops())

        context_parallel = self.config.context_parallel_size > 1

        if not context_parallel:
            # TP-only: one PartitionFuser per nano-batch
            for i in range(len(self.tp_comms)):
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
            # CP mode: three fusers covering the attention pipeline
            if len(self.cp_comms) == 0:
                return

            # QKV ops: BDA, LN, linear_qkv, qkv_postprocess, rotary_embed
            qkv_comp_ops = comp_ops[:5]
            qkv_ar_fuser = QKVPartitionFuser(
                ops=qkv_comp_ops,
                comm_op_fwd=(
                    self.tp_comms[0] if not self.is_first_layer else None
                ),
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

            # Attention + OProj ops: core_attn, linear_proj
            ao_comp_ops = comp_ops[5:]
            comm_op_fwd = [self.cp_comms[0][1], self.tp_comms[1]]
            comm_op_bwd = [
                self.cp_comms[1][1],
                self.cp_comms[0][0],
                self.cp_comms[0][1],
                self.tp_comms[1],
            ]
            ao_fuser = AttnOprojPartitionFuser(
                ops=ao_comp_ops,
                comm_ops_fwd=comm_op_fwd,
                comm_ops_bwd=comm_op_bwd,
                fuse_ops=False,
            )

            self.attention_fusers = [qkv_ar_fuser, qkv_ag_fuser, ao_fuser]

    def _build_mlp_fusers(self) -> None:
        """Build MLP partition fusers (one PartitionFuser per nano-batch)."""
        assert len(self.tp_comms) == 2, "tp_comms not initialised"

        # MLP partition ops: BDA -> LayerNorm -> FC1 -> Activation -> FC2
        comp_ops = [self.mlp_bda, self.pre_mlp_layernorm]
        comp_ops.extend(self.mlp.get_compute_ops())

        for i in range(len(self.tp_comms)):
            fwd_comm = self.tp_comms[i]
            bwd_comm = self.tp_comms[i]
            if self.is_last_layer and i == 1:
                bwd_comm = None
            self.mlp_fusers.append(
                PartitionFuser(
                    ops=comp_ops,
                    comm_op_fwd=fwd_comm,
                    comm_op_bwd=bwd_comm,
                    fuse_ops=False,
                    is_last_mlp=self.is_last_layer and i == 1,
                )
            )

    # =================================================================
    # Forward methods for nano-batch interleaved execution
    # =================================================================

    def forward_attention(
        self,
        batch_idx: int,
        hidden_states: Union[Tensor, Tuple[Tensor, Tensor]],
        residual: Union[Tensor, Tuple[Tensor, Tensor]] = None,
        comm_hidden_states: Optional[Tuple] = None,
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
    ):
        """Forward pass through the attention part of this layer.

        Called per nano-batch by TransformerBlock in the interleaved pattern.

        For TP-only mode:
            Args:
                batch_idx: 1 or 2 (nano-batch index, 1-based)
                hidden_states: raw tensor (first layer) or (tensor, bias) tuple
                residual: residual tensor
                comm_hidden_states: (tensor, bias) from other nano-batch's AllReduce
                ...standard attention kwargs...

            Returns:
                (output_hidden_states, output_residual, allreduce_output, context)
                where output_hidden_states = (tensor, bias)
                      allreduce_output = (tensor, bias) or None

        For CP mode (batch_idx=1 handles both nano-batches):
            Args:
                hidden_states: (h1, h2) tuple for first layer, or (tensor, bias)
                residual: (residual_1, residual_2) tuple
                comm_hidden_states: (tensor, bias) from other nano-batch

            Returns:
                ((out_1, bias_1), (out_2, bias_2), residual_1, residual_2)
        """
        assert context is None, "Cross-attention context is not supported"

        context_parallel = self.config.context_parallel_size > 1

        if not context_parallel:
            return self._forward_attention_tp(
                batch_idx=batch_idx,
                hidden_states=hidden_states,
                residual=residual,
                comm_hidden_states=comm_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )
        else:
            return self._forward_attention_cp(
                hidden_states=hidden_states,
                residual=residual,
                comm_hidden_states=comm_hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )

    def _forward_attention_tp(
        self,
        batch_idx: int,
        hidden_states,
        residual,
        comm_hidden_states,
        rotary_pos_emb,
        attention_mask,
    ):
        """Attention forward for TP-only mode."""
        if self.is_first_layer:
            bias = None
        else:
            hidden_states, bias = hidden_states

        output, output_bias, output_residual, allreduce_output = self.attention_fusers[
            batch_idx - 1
        ](
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_input=(
                comm_hidden_states[0] if comm_hidden_states is not None else None
            ),
            **_get_attn_fuser_comm_kwargs(self.config),
        )

        output_hidden_states = (output, output_bias)
        allreduce_output = (
            (allreduce_output, comm_hidden_states[1])
            if comm_hidden_states is not None
            else None
        )
        return output_hidden_states, output_residual, allreduce_output, None

    def _forward_attention_cp(
        self,
        hidden_states,
        residual,
        comm_hidden_states,
        rotary_pos_emb,
        attention_mask,
    ):
        """Attention forward for CP mode (processes both nano-batches)."""
        if self.is_first_layer:
            hidden_states_1, hidden_states_2 = hidden_states
            bias_1, bias_2 = None, None
            comm_input_1 = hidden_states_2
        else:
            hidden_states_1, bias_1 = hidden_states
            comm_input_1, bias_2 = comm_hidden_states

        residual_1, residual_2 = residual

        qkv_ar_fuser, qkv_ag_fuser, ao_fuser = self.attention_fusers

        # Partition 1: QKV + AllReduce (TP)
        query_1, key_1, value_1, residual_1, allreduce_output_1 = qkv_ar_fuser(
            hidden_states=hidden_states_1,
            bias=bias_1,
            residual=residual_1,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_input=comm_input_1,
            **_get_attn_fuser_comm_kwargs_cp(
                self.config, "qkv_ar", self.is_first_layer
            ),
        )
        hidden_states_2 = allreduce_output_1

        # Partition 2: QKV + AllGather (CP)
        query_2, key_2, value_2, residual_2 = qkv_ag_fuser(
            hidden_states=hidden_states_2,
            bias=bias_2,
            residual=residual_2,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            comm_key=key_1,
            comm_value=value_1,
            **_get_attn_fuser_comm_kwargs_cp(
                self.config, "qkv_ag", self.is_first_layer
            ),
        )

        # Partition 3: Attention + OProj (AllGather CP + AllReduce TP)
        out_1, out_2, bias_1, bias_2 = ao_fuser(
            query_1=query_1,
            query_2=query_2,
            comm_key=key_2,
            comm_value=value_2,
            **_get_attn_fuser_comm_kwargs_cp(
                self.config, "ao", self.is_first_layer
            ),
        )

        return (out_1, bias_1), (out_2, bias_2), residual_1, residual_2

    def forward_mlp(
        self,
        batch_idx: int,
        hidden_states: Tuple[Tensor, Optional[Tensor]],
        residual: Tensor,
        comm_hidden_states: Optional[Tuple] = None,
    ):
        """Forward pass through the MLP part of this layer.

        Called per nano-batch by TransformerBlock in the interleaved pattern.

        Args:
            batch_idx: 1 or 2 (nano-batch index, 1-based)
            hidden_states: (tensor, bias) from attention output
            residual: residual tensor
            comm_hidden_states: (tensor, bias) from other nano-batch's AllReduce

        Returns:
            ((output, output_bias), output_residual, (allreduce_output, prev_bias))
        """
        hidden_states, bias = hidden_states

        output, output_bias, output_residual, allreduce_output = self.mlp_fusers[
            batch_idx - 1
        ](
            hidden_states=hidden_states,
            bias=bias,
            residual=residual,
            comm_input=comm_hidden_states[0],
            **_get_mlp_fuser_comm_kwargs(self.config),
        )

        return (
            (output, output_bias),
            output_residual,
            (allreduce_output, comm_hidden_states[1]),
        )

    def forward(
        self,
        hidden_states: Tensor,
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
    ):
        """Standard Megatron-compatible forward (not used in Kareus interleaved mode).

        Kareus's TransformerBlock calls ``forward_attention`` and ``forward_mlp``
        directly for each nano-batch.  This method is provided for interface
        compatibility but raises NotImplementedError.
        """
        raise NotImplementedError(
            "Use forward_attention() and forward_mlp() for Kareus's "
            "nano-batch interleaved execution. The standard forward() is not "
            "supported because Kareus requires explicit nano-batch scheduling."
        )

    # =================================================================
    # Operator access (for future TensorGraph / partition building)
    # =================================================================

    def get_all_operators(self) -> List:
        """Return all operators in execution order.

        Order: [attn_bda, attn_residual_fork, input_ln,
                qkv, qkv_post, rotary, core_attn, proj,
                mlp_bda, mlp_residual_fork, pre_mlp_ln,
                fc1, activation, fc2]

        ResidualForkOp sits before each LayerNorm so that:
          Forward:  fork x into (x_main→LN, x_copy→residual for next BDA)
          Backward: accumulate grad_main + grad_residual at the fork point
        """
        ops: List = []
        # Attention partition
        ops.append(self.self_attn_bda)
        ops.append(self.attn_residual_fork)
        ops.append(self.input_layernorm)
        ops.extend(self.self_attention.get_compute_ops())
        # MLP partition
        ops.append(self.mlp_bda)
        ops.append(self.mlp_residual_fork)
        ops.append(self.pre_mlp_layernorm)
        ops.extend(self.mlp.get_compute_ops())
        return ops

    # =================================================================
    # Persistent output helpers (for comm tensor pre-allocation)
    # =================================================================

    def get_attention_persistent_outputs_fwd(self, batch_idx: int):
        """Get persistent forward output tensors for the attention module."""
        return self.self_attention.get_persistent_outputs_fwd()[batch_idx - 1]

    def get_attention_persistent_outputs_bwd(self, batch_idx: int):
        """Get persistent backward output tensors for the attention module."""
        return self.self_attention.get_persistent_outputs_bwd()[batch_idx - 1]

    def get_mlp_persistent_outputs_fwd(self, batch_idx: int):
        """Get persistent forward output tensors for the MLP module."""
        return self.mlp.get_persistent_outputs_fwd()[batch_idx - 1]

    def get_mlp_persistent_outputs_bwd(self, batch_idx: int):
        """Get persistent backward output tensors for the MLP module."""
        return self.mlp.get_persistent_outputs_bwd()[batch_idx - 1]

    # =================================================================
    # State dict
    # =================================================================

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for this transformer layer."""
        sharded_state_dict = super().sharded_state_dict(
            prefix, sharded_offsets, metadata
        )
        prefixed_map = {
            f"{prefix}{k}": f"{prefix}{v}"
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict
