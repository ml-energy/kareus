#!/usr/bin/env python
"""Verify that bf16 GEMM (torch.matmul / raw cuBLAS) produces identical results
regardless of batch size.

cuBLAS selects different algorithms for different M dimensions (M = seq * batch),
which causes different fp32 accumulation order along the K dimension.  With bf16
inputs and fp32 accumulation on Tensor Cores, this leads to different bf16
rounding at the output — affecting ~27% of elements with max_diff = 1 bf16 ULP.

Expected usage:
    python tests/partitions/test_gemm_batch_consistency.py
"""

import argparse

import torch


def run_gemm_check(args: argparse.Namespace) -> None:
    if args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be even")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    sq = args.seq_len
    b_full = args.batch_size
    b_half = b_full // 2
    hidden = args.hidden_size
    out_features = args.out_features

    W = torch.randn(out_features, hidden, device=device, dtype=dtype)
    x = torch.randn(sq, b_full, hidden, device=device, dtype=dtype)

    full_2d = x.reshape(-1, hidden).contiguous()
    half_2d = x[:, :b_half, :].reshape(-1, hidden).contiguous()
    total = sq * b_half * out_features

    print("=== GEMM Batch Consistency Check ===")
    print(f"  Config: seq={sq}, batch_full={b_full}, batch_half={b_half}, "
          f"hidden={hidden}, out={out_features}")
    print(f"  Full M={sq * b_full}, Half M={sq * b_half}, K={hidden}, N={out_features}")
    print(f"  W dtype: {W.dtype}")
    print(f"  torch.backends.cuda.matmul.allow_tf32 = "
          f"{torch.backends.cuda.matmul.allow_tf32}")

    with torch.no_grad():
        print("\n--- Full-batch vs Half-batch GEMM (torch.matmul) ---")

        out_full = torch.matmul(full_2d, W.T).reshape(sq, b_full, out_features)
        out_half = torch.matmul(half_2d, W.T).reshape(sq, b_half, out_features)
        diff = (out_full[:, :b_half] - out_half).abs()
        nz = (diff > 0).sum().item()
        print(f"  [torch.matmul] max_diff={diff.max().item():.6e}, "
              f"nonzero={nz}/{total} ({100 * nz / total:.1f}%)")

    print("\n[DONE] GEMM batch consistency check complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gemm_check(args)
