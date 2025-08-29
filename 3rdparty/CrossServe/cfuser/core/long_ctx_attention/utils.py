import torch
from typing import Optional, Tuple, List
import torch.nn.functional as F


@torch.compile(dynamic=True)
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def compute_split_sizes(
    current_rank: int,
    ranks_mlp: List[int],
    ranks_attn: List[int],
    # ranks_ulysses: List[int],
    ulysses_world_size: int,
    # ranks_ring: List[int],
    ring_world_size: int,
    scatter_idx: int,
    gather_idx: int,
) -> Tuple[List[int], List[int]]:
    """
    Given the current rank, ranks_mlp, ranks_attn, ranks_ulysses, ranks_ring, compute the split sizes for the input and output tensors.
    """

    assert len(ranks_attn) == ulysses_world_size * ring_world_size

    if scatter_idx == 2 and gather_idx == 1:
        # list version
        mlp_size = len(ranks_mlp)
        # ring_attn_world_size = len(ranks_ring)
        # ulysses_world_size = len(ranks_ulysses)

        input_split_size_list = [0] * len(ranks_mlp)
        ring_merge_size = mlp_size // ring_world_size
        ring_idx = current_rank // ring_merge_size
        for idx in ranks_attn[ring_idx * ulysses_world_size : ring_idx * ulysses_world_size + ulysses_world_size]:
            input_split_size_list[idx] = 1
        output_split_size_list = [0] * len(ranks_mlp)
        if current_rank in ranks_attn:
            ring_index = ranks_attn.index(current_rank) // ulysses_world_size
            output_split_size_list[ring_index * ring_merge_size : ring_index * ring_merge_size + ring_merge_size] = [
                1
            ] * ring_merge_size

        return input_split_size_list, output_split_size_list

    elif scatter_idx == 1 and gather_idx == 2:
        ring_merge_size = len(ranks_mlp) // ring_world_size
        # ulysses_world_size = len(ranks_ulysses)

        # list version
        input_split_size_list = [0] * len(ranks_mlp)
        if current_rank in ranks_attn:
            ring_idx = ranks_attn.index(current_rank) // ulysses_world_size
            # logger.info(f"[rank {current_rank}] ranks mlp list: {ranks_mlp[ring_idx * ring_merge_size: ring_idx * ring_merge_size + ring_merge_size]}, ring_idx: {ring_idx}")
            for idx in ranks_mlp[ring_idx * ring_merge_size : ring_idx * ring_merge_size + ring_merge_size]:
                input_split_size_list[idx] = 1

        output_split_size_list = [0] * len(ranks_mlp)
        ring_idx = current_rank // ring_merge_size
        # ulysses_idx_list = [ranks_attn[i * ring_attn_world_size + ring_idx] for i in range(ulysses_world_size)]
        ulysses_idx_list = ranks_attn[
            ring_idx * ulysses_world_size : ring_idx * ulysses_world_size + ulysses_world_size
        ]
        for idx in ulysses_idx_list:
            output_split_size_list[idx] = 1

        return input_split_size_list, output_split_size_list

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# Compute multi-dim lengths and offsets for all to all
def computeLengthsAndOffsets(
    split_sizes: List[int],
    tensor: torch.Tensor,
):
    assert tensor.is_contiguous(), "Tensor must be contiguous to compute correct offsets"
    row_size = tensor.numel() // tensor.size(0) if tensor.size(0) > 0 else 0
    lengths = []
    offsets = []
    offset = 0
    for i in range(len(split_sizes)):
        length = split_sizes[i] * row_size
        lengths.append(length)
        offsets.append(offset)
        offset += length
    return lengths, offsets
