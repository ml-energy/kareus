# Plan: Merge BasicLinear + Bias into a Single BasicOperation

## Context

`TELinearOp` extends `Linear(FusedOperation)` which contains two separate `BasicOperation`s: `BasicLinear` and `Bias`. The Kareus partition system (`forward_partition.py:91`, `backward_partition.py:90`) calls `fuser_forward`/`fuser_backward` on each `ComputeOp.operator`, expecting a single atomic operation. But `TELinearOp` as a `FusedOperation` doesn't have the right `fuser_forward`/`fuser_backward` — only its constituent basic ops do. We need to merge them into one `BasicOperation`.

## Architecture Change

```
BEFORE:                                    AFTER:
FusedOperation                             BasicOperation
  └── Linear                                 └── BasicLinear (unchanged)
        basic_ops: [BasicLinear, Bias]             └── BasicLinearBias (NEW)
        └── TELinearOp                                   └── Linear (rewritten)
                                                               └── TELinearOp (minimal changes)
```

## Steps

### Step 1: Create `BasicLinearBias`
**File:** `kareus/transformer_engine/pytorch/ops/basic/basic_linear_bias.py` (NEW)

A `BasicOperation` that inherits from `BasicLinear` and adds bias handling:

- **Constructor**: All `BasicLinear` kwargs + `has_bias`, `apply_bias`, `return_bias`
  - Creates `self.bias` parameter (shape `local_out_features`) if `has_bias`
  - Sets `num_extra_outputs = 1` if `return_bias and has_bias`, else `0`

- **`op_forward(ctx, input_, prev_op, next_op, batch_idx=0)`**:
  - Reuses all FP8/quantizer/persistent-output logic from `BasicLinear.op_forward`
  - Calls `BasicLinear._functional_forward(bias=self.bias)` when `apply_bias=True` (cuBLAS fuses GEMM+bias)
  - Calls `BasicLinear._functional_forward(bias=None)` when `return_bias=True`
  - Returns `(output, self.bias)` when `return_bias`, else `(output, None)`
  - Reference: `forward_linear_bias_activation.py:110-136`

- **`op_backward(ctx, grad_output, grad_bias=None)`**:
  - Calls `BasicLinear._functional_backward()` → `(grad_input, grad_weight)`
  - When `apply_bias`: `grad_bias = grad_output` (matches `bias.py:160` pattern)
  - When `return_bias`: `grad_bias` comes from upstream via parameter
  - Returns `(grad_input, (grad_weight, grad_bias))` or `(grad_input, (grad_weight,))`

- **`fuser_forward`**: Wraps `op_forward` for partition system
  - Returns `(main_out, [(bias,)])` or `(main_out, [()])`

- **`fuser_backward`**: Wraps `op_backward` for partition system
  - Extracts `grad_bias` from `basic_op_grad_extra_outputs[0]` when `return_bias`
  - Returns `(grad_input, [grad_params], [()])`

### Step 2: Update exports
**File:** `kareus/transformer_engine/pytorch/ops/basic/__init__.py`

Add `from .basic_linear_bias import BasicLinearBias`.

### Step 3: Rewrite `Linear`
**File:** `kareus/transformer_engine/pytorch/ops/linear.py`

Change from `FusedOperation` wrapping `[BasicLinear, Bias]` to `Linear(BasicLinearBias)`:

- `__init__`: Perform TP canonicalization via `BasicLinear._canonicalize_tensor_parallelism()`, then call `super().__init__()` with local dimensions and `tensor_parallel_mode=None` (TP already applied)
- Remove all `basic_ops` references, proxy properties, and `FusedOperation` imports
- `weight` and `bias` are now direct `nn.Parameter` attrs inherited from `BasicLinear`/`BasicLinearBias`
- `fuser_forward`/`fuser_backward` inherited from `BasicLinearBias`

### Step 4: Update `TELinearOp`
**File:** `kareus/megatron/core/extensions/ops/te_linear.py`

Minimal changes:
- `forward()`: Change from `super().forward(x, basic_op_kwargs=[{"batch_idx": batch_idx}, {}])` to `super().forward(x, batch_idx=batch_idx)` (single op, not two)
  - `BasicOperation.forward` passes kwargs to `OperationFuser` → `fuser_forward` → `op_forward`
  - Handle return type: tuple when `return_bias`, single tensor otherwise
- No changes to `_handle_cpu_initialization` or `_set_gradient_attributes` (they use `self.weight`/`self.bias`/`self.parameters()` which still work)
- No changes to `TEColumnParallelLinearOp`/`TERowParallelLinearOp` partition methods

### Step 5: No changes to `ForwardLinearBiasActivation`
**File:** `kareus/transformer_engine/pytorch/ops/fused/forward_linear_bias_activation.py`

This file's fusion pass (`fuse_forward_linear_bias_activation`) scans for consecutive `BasicLinear` + `Bias` ops. Since `Linear` no longer produces separate ops, this function simply won't match anything. It remains as dead code for backward compatibility with upstream TE.

## Key Design Details

| Aspect | Old (FusedOperation) | New (BasicLinearBias) |
|--------|---------------------|----------------------|
| `weight` | `basic_ops[0].weight` | Direct `nn.Parameter` |
| `bias` | `basic_ops[1].bias` | Direct `nn.Parameter` |
| `state_dict` keys | `basic_ops.0.weight`, `basic_ops.1.bias` | `weight`, `bias` |
| `fuser_forward` | Not implemented (was on sub-ops) | Implemented, called by partition system |
| `forward()` | `FusedOperation.forward` with `basic_op_kwargs` list | `BasicOperation.forward` with `**kwargs` |
| `parameters()` | Yields from BasicLinear + Bias modules | Yields weight + bias directly |

**State dict key change**: `sharded_state_dict` in `TEColumnParallelLinearOp`/`TERowParallelLinearOp` already uses `state_dict(prefix='', keep_vars=True)` and maps to `'weight'`/`'bias'` keys explicitly, so checkpoint compatibility is preserved.

**Gradient matching**: `fuser_backward` returns `[(grad_weight, grad_bias)]` matching `parameters()` order `[weight, bias]`. When `has_bias=False`: returns `[(grad_weight,)]` matching `[weight]`.

## Verification

1. Run `python tests/partitions/test_ops_consistency.py` — tests TELinearOp consistency
2. Run `python tests/llama/kareus_gpt_pretraining.py` — end-to-end training
3. Verify forward output matches between old and new implementation
4. Verify backward gradients (weight + bias) match
