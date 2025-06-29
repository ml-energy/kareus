"""Rotary Embedding operation following the BasicOperation pattern."""

import torch
from typing import Optional, Tuple, Union, Callable

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
# from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.inference.contexts import BaseInferenceContext
from kareus.megatron.core.models.common.embedding.rope_utils import apply_rotary_pos_emb, apply_rotary_pos_emb_backward

class RotaryEmbeddingOp(BasicOperation):
    """Rotary Embedding as a BasicOperation
    
    This operation applies rotary positional embeddings to query and key tensors.
    
    Parameters
    ----------
    config : TransformerConfig
        Transformer configuration containing RoPE settings
    inference_context : Optional, default = None
        Inference context for handling different batching modes
    """

    # RotaryEmbedding has 2 extra inputs: key and rotary_pos_emb (query is the main input)
    # RotaryEmbedding has 1 extra outputs: key (query is the main output)
    num_extra_inputs: int = 2
    num_extra_outputs: int = 1

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
        key: torch.Tensor,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None, 
        inference_context: Optional[BaseInferenceContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for rotary embedding application.
        
        Args:
            ctx: Operation context for saving state
            input_: Query tensor to apply rotary embedding to
            key: Key tensor to apply rotary embedding to
            rotary_pos_emb: Rotary position embedding(s)
            packed_seq_params: Parameters for packed sequence format
            inference_context: Inference context for different batching modes
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Modified query and key tensors
        """
        if inference_context is not None:
            raise NotImplementedError("Inference context not supported")

        query = input_

        ctx.duplicate_rotary_pos_emb = False
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            ctx.duplicate_rotary_pos_emb = True
            rotary_pos_emb = (rotary_pos_emb,) * 2
        else:
            raise ValueError("rotary_pos_emb cannot be a tuple of two tensors")
        
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb
        
            # Handle packed sequence parameters
            if packed_seq_params is not None:
                raise NotImplementedError("Packed sequence parameters not supported")
                # if packed_seq_params.cu_seqlens_q_padded is not None:
                #     cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                # else:
                #     cu_seqlens_q = packed_seq_params.cu_seqlens_q
                # if packed_seq_params.cu_seqlens_kv_padded is not None:
                #     cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                # else:
                #     cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            # Create contexts for saving state needed for backward pass
            query_ctx = OperationContext()
            key_ctx = OperationContext()

            # Apply rotary embedding to query
            if q_pos_emb is not None:
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query_ctx, query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                    )
                # else:
                #     query = self.inference_context.apply_rotary_emb_query(
                #         query, q_pos_emb, self.config, cu_seqlens_q
                #     )

            # Apply rotary embedding to key
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key_ctx, key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                )

            # Save contexts and metadata for backward pass
            assert q_pos_emb is not None and k_pos_emb is not None
            to_save = []
            for ctx_ in [query_ctx, key_ctx]:
                range_start = len(to_save)
                if ctx_.to_save is not None:
                    to_save.extend(ctx_.to_save)
                range_end = len(to_save)
                ctx_.to_save = None
                ctx_._saved_tensors_range = (range_start, range_end)
            ctx.save_for_backward(*to_save)
            ctx.query_ctx = query_ctx
            ctx.key_ctx = key_ctx
        
        return query, key

    def op_backward(
        self,
        ctx: OperationContext,
        grad_query: torch.Tensor,
        grad_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Backward pass for rotary embedding application.
        
        Args:
            ctx: Operation context with saved tensors and contexts
            grad_query: Gradient w.r.t. query tensor
            grad_key: Gradient w.r.t. key tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Gradients w.r.t. input query, key, and rotary_pos_emb
        """
        
        # Initialize variables
        grad_query_input = grad_query
        grad_key_input = grad_key
        grad_rotary_pos_emb = None
        query_rope_applied = False
        key_rope_applied = False

        for ctx_ in [ctx.query_ctx, ctx.key_ctx]:
            ctx_.saved_tensors = ctx.saved_tensors[slice(*ctx_._saved_tensors_range)]
            ctx_._saved_tensors_range = None

        # Apply backward pass for rotary embeddings if they were applied in forward
        if hasattr(ctx, 'query_ctx'):
            query_rope_applied = True
            grad_query_input, grad_q_freqs = apply_rotary_pos_emb_backward(
                ctx.query_ctx, grad_query
            )
            
        if hasattr(ctx, 'key_ctx'):
            key_rope_applied = True
            grad_key_input, grad_k_freqs = apply_rotary_pos_emb_backward(
                ctx.key_ctx, grad_key
            )
            
        # Combine frequency gradients if both query and key had rotary embeddings applied
        if query_rope_applied and key_rope_applied:
            # If the same frequency tensor was used for both, combine gradients
            if ctx.duplicate_rotary_pos_emb:
                grad_rotary_pos_emb = grad_q_freqs + grad_k_freqs
            else:
                # Different frequency tensors were used
                grad_rotary_pos_emb = (grad_q_freqs, grad_k_freqs)
        elif query_rope_applied:
            grad_rotary_pos_emb = (grad_q_freqs, None)
        elif key_rope_applied:
            grad_rotary_pos_emb = (None, grad_k_freqs)

        return grad_query_input, grad_key_input, grad_rotary_pos_emb

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, any]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor]]]:
        """Fuser forward pass with extra outputs.
        
        Returns:
            main_output: Query tensor with rotary embedding applied
            extra_outputs: List containing key tensor with rotary embedding applied
        """

        key, rotary_pos_emb = basic_op_extra_inputs[0]
        
        # Extract rotary embedding parameters from kwargs
        kwargs = basic_op_kwargs[0].copy()
        kwargs['key'] = key
        kwargs['rotary_pos_emb'] = rotary_pos_emb
        
        # Forward pass
        query_out, key_out = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **kwargs,
        )
        
        return query_out, [(key_out,)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[torch.Tensor]],
    ]:
        """Fuser backward pass handling extra output gradients.
        
        Args:
            basic_op_ctxs: Operation contexts
            grad_output: Gradient w.r.t. main output (query)
            basic_op_grad_extra_outputs: Gradients w.r.t. extra outputs (key)
            
        Returns:
            grad_query_input: Gradient w.r.t. query input
            grad_extra_inputs: List containing gradient w.r.t. key input and rotary_pos_emb
            grad_extra_outputs: Gradients w.r.t. extra outputs
        """
        
        # Get gradient w.r.t. key
        grad_key, = basic_op_grad_extra_outputs[0]
        
        # Perform backward pass
        grad_query_input, grad_key_input, grad_rotary_pos_emb = self.op_backward(
            basic_op_ctxs[0], grad_output, grad_key
        )
        
        return grad_query_input, [()], [(grad_key_input, grad_rotary_pos_emb)]


def create_rotary_embedding_op(
    config: TransformerConfig,
) -> RotaryEmbeddingOp:
    """Factory function to create a rotary embedding operation.
    
    Args:
        config: Transformer configuration containing RoPE settings
        
    Returns:
        RotaryEmbeddingOp: Configured rotary embedding operation
    """
    return RotaryEmbeddingOp(
        config=config,
    ) 