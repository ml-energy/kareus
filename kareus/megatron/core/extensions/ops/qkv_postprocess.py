"""QKV Post-Processing operation following the BasicOperation pattern."""

import torch
from typing import List, Optional, Tuple, Callable

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext
from kareus.megatron.core.partitions.tensor_graph import (
    Channel,
    PartitionableOperator,
)
from kareus.megatron.core.extensions.ops.utils import merge_sub_contexts, restore_sub_contexts


class QKVPostProcessOp(BasicOperation, PartitionableOperator):
    """QKV Post-Processing as a BasicOperation

    This operation takes the output of a linear QKV layer and post-processes it to
    produce separate query, key, and value tensors with proper reshaping and
    optional layer normalization.

    Parameters
    ----------
    num_query_groups_per_partition : int
        Number of query groups per partition
    num_attention_heads_per_partition : int
        Number of attention heads per partition
    hidden_size_per_attention_head : int
        Hidden size per attention head
    q_layernorm : Optional[BasicOperation], default = None
        Query layer normalization module
    k_layernorm : Optional[BasicOperation], default = None
        Key layer normalization module
    run_tests_fn : Optional[callable], default = None
        Function to run consistency tests
    test_mode : bool, default = False
        Whether to run tests during forward pass
    """

    # QKVPostProcess has 2 extra outputs: key and value (query is the main output)
    num_extra_outputs: int = 2

    def get_output_channels(self) -> List[Channel]:
        return [Channel(0, "main"), Channel(1, "key"), Channel(2, "value")]

    def __init__(
        self,
        num_query_groups_per_partition: int,
        num_attention_heads_per_partition: int,
        hidden_size_per_attention_head: int,
        q_layernorm: Optional[BasicOperation] = None,
        k_layernorm: Optional[BasicOperation] = None,
        run_tests_fn: Optional[Callable] = None,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_query_groups_per_partition = num_query_groups_per_partition
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.q_layernorm = q_layernorm
        self.k_layernorm = k_layernorm
        
        self.run_tests_fn = run_tests_fn
        self.test_mode = test_mode

    def op_forward(
        self,
        ctx: OperationContext,
        mixed_qkv: torch.Tensor,
        *,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for QKV post-processing.
        
        Args:
            ctx: Operation context for saving state
            mixed_qkv: Output from linear_qkv layer with shape [sq, b, hp]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query, key, and value tensors
        """
        
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv_reshaped = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = torch.split(mixed_qkv_reshaped, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        # Apply layernorm with sub-contexts (following RotaryEmbeddingOp pattern)
        q_ln_ctx = OperationContext()
        k_ln_ctx = OperationContext()

        if self.q_layernorm is not None:
            query = self.q_layernorm.op_forward(q_ln_ctx, query)
        if self.k_layernorm is not None:
            key = self.k_layernorm.op_forward(k_ln_ctx, key)

        merge_sub_contexts(ctx, [q_ln_ctx, k_ln_ctx])
        ctx.q_ln_ctx = q_ln_ctx
        ctx.k_ln_ctx = k_ln_ctx

        if self.test_mode and self.run_tests_fn is not None:
            self.run_tests_fn()
        
        return query, key, value

    def op_backward(
        self,
        ctx: OperationContext,
        grad_query: torch.Tensor,
        grad_key: torch.Tensor,
        grad_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple]:
        """Backward pass for QKV post-processing with gradients for all outputs.
        
        Args:
            ctx: Operation context with saved tensors
            grad_query: Gradient w.r.t. query tensor [sq, b, np, hn]
            grad_key: Gradient w.r.t. key tensor [sq, b, ng, hn]
            grad_value: Gradient w.r.t. value tensor [sq, b, ng, hn]
            
        Returns:
            grad_mixed_qkv: Gradient w.r.t. input mixed_qkv tensor [sq, b, hp]
            grad_params: Tuple of layernorm weight gradients (empty when no layernorms)
        """

        restore_sub_contexts(ctx, [ctx.q_ln_ctx, ctx.k_ln_ctx])

        # Backprop through layernorms — each returns (grad_input, (grad_weight,))
        grad_params = []
        if self.q_layernorm is not None:
            grad_query, q_weight_grads = self.q_layernorm.op_backward(
                ctx.q_ln_ctx, grad_query
            )
            grad_params.extend(q_weight_grads)
        if self.k_layernorm is not None:
            grad_key, k_weight_grads = self.k_layernorm.op_backward(
                ctx.k_ln_ctx, grad_key
            )
            grad_params.extend(k_weight_grads)
        
        # [sq, b, np, hn] -> [sq, b, ng, np/ng * hn]
        grad_query_reshaped = grad_query.reshape(
            grad_query.size(0), 
            grad_query.size(1), 
            self.num_query_groups_per_partition,
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition) * self.hidden_size_per_attention_head
        )
        
        # [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn] -> [sq, b, ng, (np/ng + 2) * hn]
        grad_mixed_qkv_reshaped = torch.cat([grad_query_reshaped, grad_key, grad_value], dim=3)
        
        # [sq, b, ng, (np/ng + 2) * hn] -> [sq, b, hp]
        original_shape = grad_mixed_qkv_reshaped.size()[:-2] + (
            self.num_query_groups_per_partition * (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        grad_mixed_qkv = grad_mixed_qkv_reshaped.view(*original_shape)
        
        return grad_mixed_qkv, tuple(grad_params)

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, any]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Fuser forward pass with extra outputs.
        
        Returns:
            main_output: Query tensor
            extra_outputs: List containing (key, value) tuple
        """
        
        # Forward pass
        query, key, value = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op=basic_op_prev_ops[0],
            next_op=basic_op_next_ops[0],
            **basic_op_kwargs[0],
        )
        
        return query, [(key, value)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Fuser backward pass handling extra output gradients.
        
        Args:
            basic_op_ctxs: Operation contexts
            grad_output: Gradient w.r.t. main output (query)
            basic_op_grad_extra_outputs: Gradients w.r.t. extra outputs (key, value)
            
        Returns:
            grad_input: Gradient w.r.t. input
            grad_extra_inputs: Empty list (no extra inputs)
            grad_extra_outputs: Gradients w.r.t. extra outputs
        """
        
        grad_key, grad_value = basic_op_grad_extra_outputs[0]
        
        grad_mixed_qkv, grad_params = self.op_backward(
            basic_op_ctxs[0], grad_output, grad_key, grad_value
        )
        
        return grad_mixed_qkv, [grad_params], [()]


def create_qkv_postprocess_op(
    num_query_groups_per_partition: int,
    num_attention_heads_per_partition: int,
    hidden_size_per_attention_head: int,
    q_layernorm: Optional[BasicOperation] = None,
    k_layernorm: Optional[BasicOperation] = None,
    run_tests_fn: Optional[Callable] = None,
    test_mode: bool = False,
) -> QKVPostProcessOp:
    """Factory function to create a QKV post-processing operation.
    
    Args:
        num_query_groups_per_partition: Number of query groups per partition
        num_attention_heads_per_partition: Number of attention heads per partition
        hidden_size_per_attention_head: Hidden size per attention head
        q_layernorm: Optional query layer normalization BasicOperation
        k_layernorm: Optional key layer normalization BasicOperation
        run_tests_fn: Optional function to run consistency tests
        test_mode: Whether to run tests during forward pass
        
    Returns:
        QKVPostProcessOp: Configured QKV post-processing operation
    """
    return QKVPostProcessOp(
        num_query_groups_per_partition=num_query_groups_per_partition,
        num_attention_heads_per_partition=num_attention_heads_per_partition,
        hidden_size_per_attention_head=hidden_size_per_attention_head,
        q_layernorm=q_layernorm,
        k_layernorm=k_layernorm,
        run_tests_fn=run_tests_fn,
        test_mode=test_mode,
    )
