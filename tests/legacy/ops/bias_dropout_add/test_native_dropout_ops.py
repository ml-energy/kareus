import math
import pytest
import torch


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5])
def test_native_dropout_backward_matches_manual(p):
    for device in available_devices():
        torch.manual_seed(1234)

        input_tensor = torch.randn(8, 8, device=device, dtype=torch.float32)
        grad_output = torch.randn_like(input_tensor)

        # Forward via internal ATen op to get output and mask
        output_tensor, mask = torch.ops.aten.native_dropout(input_tensor, float(p), True)

        # Compute backward via internal ATen op
        scale = 0.0 if math.isclose(1.0 - p, 0.0) else 1.0 / (1.0 - p)
        grad_input_native = torch.ops.aten.native_dropout_backward(grad_output, mask, scale)

        # Manual expected gradient: dy * mask * scale
        grad_input_manual = grad_output * mask.to(dtype=grad_output.dtype) * scale

        assert torch.allclose(
            grad_input_native, grad_input_manual, rtol=1e-6, atol=1e-7
        ), f"Mismatch on device={device}, p={p}"


@pytest.mark.parametrize("p", [0.0, 0.2, 0.5])
def test_native_dropout_backward_matches_autograd(p):
    for device in available_devices():
        torch.manual_seed(4321)

        input_tensor = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
        grad_output = torch.randn_like(input_tensor)

        # Forward via internal ATen op to get output and mask
        output_tensor, mask = torch.ops.aten.native_dropout(input_tensor, float(p), True)

        # Build a scalar loss using a fixed upstream gradient to validate backward
        loss = (output_tensor * grad_output).sum()
        loss.backward()

        # Expected gradients
        scale = 0.0 if math.isclose(1.0 - p, 0.0) else 1.0 / (1.0 - p)
        grad_input_manual = grad_output * mask.to(dtype=grad_output.dtype) * scale

        # Native backward called directly
        grad_input_native = torch.ops.aten.native_dropout_backward(grad_output, mask, scale)

        assert torch.allclose(
            input_tensor.grad, grad_input_manual, rtol=1e-6, atol=1e-7
        ), f"Autograd mismatch on device={device}, p={p}"

        assert torch.allclose(
            input_tensor.grad, grad_input_native, rtol=1e-6, atol=1e-7
        ), f"Native backward mismatch on device={device}, p={p}"


