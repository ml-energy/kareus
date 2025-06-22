"""QKV Post-Processing operation following the BasicOperation pattern."""

import torch
from typing import Optional, Tuple, Union, Callable

from transformer_engine.pytorch.ops.op import BasicOperation, OperationContext


class QKVPostProcessOp(BasicOperation):
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
    q_layernorm : Optional[torch.nn.Module], default = None
        Query layer normalization module
    k_layernorm : Optional[torch.nn.Module], default = None
        Key layer normalization module
    run_tests_fn : Optional[callable], default = None
        Function to run consistency tests
    test_mode : bool, default = False
        Whether to run tests during forward pass
    """

    # QKVPostProcess has 2 extra outputs: key and value (query is the main output)
    num_extra_outputs: int = 2

    def __init__(
        self,
        num_query_groups_per_partition: int,
        num_attention_heads_per_partition: int,
        hidden_size_per_attention_head: int,
        q_layernorm: Optional[torch.nn.Module] = None,
        k_layernorm: Optional[torch.nn.Module] = None,
        run_tests_fn: Optional[Callable] = None,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_query_groups_per_partition = num_query_groups_per_partition
        self.num_attention_heads_per_partition = num_attention_heads_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        # self.q_layernorm = q_layernorm
        # self.k_layernorm = k_layernorm
        if q_layernorm is not None or k_layernorm is not None:
            raise NotImplementedError("q_layernorm and k_layernorm not supported")
        
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

        # # Apply layer normalization if provided
        # if self.q_layernorm is not None:
        #     query = self.q_layernorm(query)

        # if self.k_layernorm is not None:
        #     key = self.k_layernorm(key)

        # Run consistency tests if configured
        if self.test_mode and self.run_tests_fn is not None:
            self.run_tests_fn()

        # Save all tensors for backward pass
        # ctx.save_for_backward(query, key, value)
        
        return query, key, value

    # def op_backward(
    #     self,
    #     ctx: OperationContext,
    #     grad_output: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Backward pass for QKV post-processing.
        
    #     Args:
    #         ctx: Operation context with saved tensors
    #         grad_output: Gradient w.r.t. query tensor [sq, b, np, hn]
            
    #     Returns:
    #         grad_mixed_qkv: Gradient w.r.t. input mixed_qkv tensor [sq, b, hp]
    #     """
        
    #     # Retrieve saved tensors (query, key, value)
    #     query, key, value = ctx.saved_tensors
        
    #     # Initialize gradients for key and value (these would come from the attention computation)
    #     # For now, we'll assume they are zeros, but in practice they would be provided
    #     grad_key = torch.zeros_like(key)
    #     grad_value = torch.zeros_like(value)
        
    #     # Apply layer norm backward if it was used
    #     if self.q_layernorm is not None:
    #         # grad_output is w.r.t. normalized query, need to get grad w.r.t. unnormalized query
    #         # This is a simplified version - actual implementation would need proper layernorm backward
    #         grad_query = grad_output
    #     else:
    #         grad_query = grad_output
            
    #     if self.k_layernorm is not None:
    #         # Similar for key layernorm - this would be handled by the actual layernorm backward
    #         pass
        
    #     # Reshape query gradient back to grouped format
    #     # [sq, b, np, hn] -> [sq, b, ng, np/ng * hn]
    #     grad_query_reshaped = grad_query.reshape(
    #         grad_query.size(0), 
    #         grad_query.size(1), 
    #         self.num_query_groups_per_partition,
    #         (self.num_attention_heads_per_partition // self.num_query_groups_per_partition) * self.hidden_size_per_attention_head
    #     )
        
    #     # Concatenate gradients along the last dimension to reconstruct mixed_qkv gradient
    #     # [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn] -> [sq, b, ng, (np/ng + 2) * hn]
    #     grad_mixed_qkv_reshaped = torch.cat([grad_query_reshaped, grad_key, grad_value], dim=3)
        
    #     # Reshape back to original mixed_qkv shape
    #     # [sq, b, ng, (np/ng + 2) * hn] -> [sq, b, hp]
    #     original_shape = grad_mixed_qkv_reshaped.size()[:-2] + (
    #         self.num_query_groups_per_partition * (
    #             (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
    #             * self.hidden_size_per_attention_head
    #         ),
    #     )
    #     grad_mixed_qkv = grad_mixed_qkv_reshaped.view(*original_shape)
        
    #     return grad_mixed_qkv

    def op_backward(
        self,
        ctx: OperationContext,
        grad_query: torch.Tensor,
        grad_key: torch.Tensor,
        grad_value: torch.Tensor,
    ) -> torch.Tensor:
        """Backward pass for QKV post-processing with gradients for all outputs.
        
        Args:
            ctx: Operation context with saved tensors
            grad_query: Gradient w.r.t. query tensor [sq, b, np, hn]
            grad_key: Gradient w.r.t. key tensor [sq, b, ng, hn]
            grad_value: Gradient w.r.t. value tensor [sq, b, ng, hn]
            
        Returns:
            grad_mixed_qkv: Gradient w.r.t. input mixed_qkv tensor [sq, b, hp]
        """
        
        # Retrieve saved tensors (query, key, value)
        # query, key, value = ctx.saved_tensors
        
        # Reshape query gradient back to grouped format
        # [sq, b, np, hn] -> [sq, b, ng, np/ng * hn]
        grad_query_reshaped = grad_query.reshape(
            grad_query.size(0), 
            grad_query.size(1), 
            self.num_query_groups_per_partition,
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition) * self.hidden_size_per_attention_head
        )
        
        # Concatenate gradients along the last dimension to reconstruct mixed_qkv gradient
        # [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn] -> [sq, b, ng, (np/ng + 2) * hn]
        grad_mixed_qkv_reshaped = torch.cat([grad_query_reshaped, grad_key, grad_value], dim=3)
        
        # Reshape back to original mixed_qkv shape
        # [sq, b, ng, (np/ng + 2) * hn] -> [sq, b, hp]
        original_shape = grad_mixed_qkv_reshaped.size()[:-2] + (
            self.num_query_groups_per_partition * (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        grad_mixed_qkv = grad_mixed_qkv_reshaped.view(*original_shape)
        
        return grad_mixed_qkv

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
        
        # Get gradients w.r.t. key and value
        grad_key, grad_value = basic_op_grad_extra_outputs[0]
        
        # Perform backward pass with all gradients
        grad_mixed_qkv = self.op_backward(
            basic_op_ctxs[0], grad_output, grad_key, grad_value
        )
        
        return grad_mixed_qkv, [], [(grad_key, grad_value)]


def create_qkv_postprocess_op(
    num_query_groups_per_partition: int,
    num_attention_heads_per_partition: int,
    hidden_size_per_attention_head: int,
    q_layernorm: Optional[torch.nn.Module] = None,
    k_layernorm: Optional[torch.nn.Module] = None,
    run_tests_fn: Optional[Callable] = None,
    test_mode: bool = False,
) -> QKVPostProcessOp:
    """Factory function to create a QKV post-processing operation.
    
    Args:
        num_query_groups_per_partition: Number of query groups per partition
        num_attention_heads_per_partition: Number of attention heads per partition
        hidden_size_per_attention_head: Hidden size per attention head
        q_layernorm: Optional query layer normalization module
        k_layernorm: Optional key layer normalization module
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