import pytest
import torch

import sys
sys.path.append("../../../")

from kareus.transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear


def test_persistent_output():
    batch_size = 4
    seq_length = 4096
    in_features = 2048
    out_features = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    linear_persistent = BasicLinear(
        in_features=in_features,
        out_features=out_features,
        device=device,
        dtype=dtype,
        use_persistent_output=True,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    linear_non_persistent = BasicLinear(
        in_features=in_features,
        out_features=out_features,
        device=device,
        dtype=dtype,
        use_persistent_output=False
    )

    with torch.no_grad():
        linear_non_persistent.weight.copy_(linear_persistent.weight)

    print("Test 1")
    x = torch.randn(seq_length, batch_size, in_features, device=device, dtype=dtype)

    output_persistent = linear_persistent(x)
    output_non_persistent = linear_non_persistent(x)

    torch.testing.assert_close(output_persistent, output_non_persistent)

    print("Test 2")
    x2 = torch.randn(seq_length, batch_size, in_features, device=device, dtype=dtype)

    output_persistent_2 = linear_persistent(x2)
    output_non_persistent_2 = linear_non_persistent(x2)

    torch.testing.assert_close(output_persistent_2, output_non_persistent_2)

    print("All tests passed")

test_persistent_output()