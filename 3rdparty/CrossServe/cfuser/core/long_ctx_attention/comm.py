import torch
import torch.distributed as dist
from typing import Optional, List, Union, Tuple

from cfuser.logger import init_logger
from cfuser.core.long_ctx_attention.pynccl_wrapper import PyNcclCommunicator
from cfuser.msccl_comm import msccl_AlltoAllv
from .utils import compute_split_sizes

logger = init_logger(__name__)


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


def uneven_decoupled_all_to_all_4D(
    input: Union[torch.tensor, torch.Size],
    ranks_mlp: List[int],
    ranks_attn: List[int],
    ranks_ulysses: List[int],
    ranks_ring: List[int],
    group_mlp: dist.ProcessGroup,
    dtype: torch.dtype,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    async_op: bool = False,
    stream: Optional[torch.cuda.Stream] = None,
    custom_backend: Optional[str] = "torch",
) -> Union[torch.tensor, Tuple[torch.tensor, List[dist.Work]]]:
    """
    Disaggregated all-to-all for QKV
    On top of uneven_all_to_all_4D, supports both ulysses and ring_attn sharding
    """

    assert custom_backend in [
        "python",
        "nccl",
        "torch",
        "mscclpp",
    ], f"custom_backend must be one of ['python', 'nccl', 'torch', 'mscclpp'], got {custom_backend}"
    if custom_backend == "torch":
        all_to_all_impl = dist.all_to_all_single
    elif custom_backend == "python":
        all_to_all_impl = custom_all_to_all_single
    elif custom_backend == "nccl":
        all_to_all_impl = custom_all_to_all_single_nccl
    elif custom_backend == "mscclpp":
        all_to_all_impl = msccl_AlltoAllv

    if scatter_idx == 2 and gather_idx == 1:
        assert isinstance(input, torch.Tensor), "input must be a tensor"
        assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P_send, hc, hs) output: (bs, seqlen, hc/P_recv, hs)
        bs, shard_seqlen, hc, hs = input.shape

        current_rank = dist.get_global_rank(group_mlp, group_mlp.rank())
        # logger.info(f"[rank {current_rank}] input shape: {input.shape}")
        mlp_size = len(ranks_mlp)
        ring_attn_world_size = len(ranks_ring)
        ulysses_world_size = len(ranks_ulysses)
        ring_merge_size = mlp_size // ring_attn_world_size

        shard_hc = hc // ulysses_world_size

        input_t = (
            input.reshape(bs, shard_seqlen, ulysses_world_size, shard_hc, hs).transpose(0, 2).contiguous()
        )  # (ulysses_world_size, shard_seqlen, bs, shard_hc, hs)

        # list version
        input_split_size_list, output_split_size_list = compute_split_sizes(
            current_rank=current_rank,
            ranks_mlp=ranks_mlp,
            ranks_attn=ranks_attn,
            # ranks_ulysses,
            # ranks_ring,
            ulysses_world_size=ulysses_world_size,
            ring_world_size=ring_attn_world_size,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
        )

        output_shape = list(input_t.shape)
        output_shape[0] = sum(output_split_size_list)
        output = torch.empty(output_shape, device=input_t.device, dtype=input_t.dtype)

        if custom_backend == "mscclpp":
            msccl_AlltoAllv(
                input=input_t,
                output=output,
                current_rank=current_rank,
                ranks_mlp=ranks_mlp,
                ranks_attn=ranks_attn,
                ranks_ulysses=ranks_ulysses,
                ranks_ring=ranks_ring,
                stream=torch.cuda.current_stream() if stream is None else stream,
                sm_num=len(ranks_mlp) - 1,
                block_size=256,
                nranks=len(ranks_mlp),
                scatter_idx=scatter_idx,
                gather_idx=gather_idx,
            )
        else:
            handle = all_to_all_impl(
                output,
                input_t,
                output_split_sizes=output_split_size_list,
                input_split_sizes=input_split_size_list,
                group=group_mlp,
                async_op=async_op,
            )

        if not async_op:
            if output_shape[0] != 0:
                # logger.info(f"[rank {current_rank}] output shape: {output.shape}")
                # logger.info(f"[rank {current_rank}] shard_seqlen: {shard_seqlen}, ring_merge_size: {ring_merge_size}, bs: {bs}, shard_hc: {shard_hc}")
                output = output.reshape(shard_seqlen * ring_merge_size, bs, shard_hc, hs)
                output = output.transpose(0, 1).contiguous()
                # logger.info(f"[rank {current_rank}] final output shape: {output.shape}")
            else:
                output = None

        return (output, handle) if async_op else output

    elif scatter_idx == 1 and gather_idx == 2:
        current_rank = dist.get_global_rank(group_mlp, group_mlp.rank())
        assert len(ranks_mlp) == dist.get_world_size(
            group_mlp
        ), f"ranks_mlp must have the same length as group_mlp, got {len(ranks_mlp)} and {dist.get_world_size(group_mlp)}"

        ring_merge_size = len(ranks_mlp) // len(ranks_ring)
        ulysses_world_size = len(ranks_ulysses)
        ring_attn_world_size = len(ranks_ring)

        if isinstance(input, torch.Tensor):
            assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"
            bs, shard_seqlen, shard_hc, hs = input.shape

            input_t = (
                input.reshape(bs, ring_merge_size, shard_seqlen // ring_merge_size, shard_hc, hs)
                .permute(1, 3, 2, 0, 4)
                .contiguous()
            )  # (ring_merge_size, shard_hc, shard_seqlen // ring_merge_size, bs, hs)
        elif isinstance(input, torch.Size):
            bs, shard_seqlen, shard_hc, hs = input
            input_t = torch.empty(
                (0, ring_merge_size, shard_seqlen // ring_merge_size, shard_hc, hs),
                device=f"cuda:{current_rank}",
                dtype=dtype,
            )
        else:
            raise RuntimeError("input must be a tensor or a torch.Size")

        # list version
        input_split_size_list, output_split_size_list = compute_split_sizes(
            current_rank=current_rank,
            ranks_mlp=ranks_mlp,
            ranks_attn=ranks_attn,
            # ranks_ulysses,
            # ranks_ring,
            ulysses_world_size=ulysses_world_size,
            ring_world_size=ring_attn_world_size,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
        )

        # logger.info(f"[rank {current_rank}] input_split_size_list: {input_split_size_list}")
        # logger.info(f"[rank {current_rank}] output_split_size_list: {output_split_size_list}")
        # dist.barrier(group=group_mlp)

        output_shape = [ulysses_world_size, shard_hc, shard_seqlen // ring_merge_size, bs, hs]
        output = torch.empty(output_shape, device=input_t.device, dtype=dtype)

        if custom_backend == "mscclpp":
            msccl_AlltoAllv(
                input=input_t,
                output=output,
                current_rank=current_rank,
                ranks_mlp=ranks_mlp,
                ranks_attn=ranks_attn,
                ranks_ulysses=ranks_ulysses,
                ranks_ring=ranks_ring,
                stream=torch.cuda.current_stream() if stream is None else stream,
                sm_num=len(ranks_mlp) - 1,
                block_size=256,
                nranks=len(ranks_mlp),
                scatter_idx=scatter_idx,
                gather_idx=gather_idx,
            )
        else:
            handle = all_to_all_impl(
                output,
                input_t,
                output_split_sizes=output_split_size_list,
                input_split_sizes=input_split_size_list,
                group=group_mlp,
                async_op=async_op,
            )

        if not async_op:
            assert output.shape == (ulysses_world_size, shard_hc, shard_seqlen // ring_merge_size, bs, hs)
            output = (
                output.reshape(ulysses_world_size * shard_hc, shard_seqlen // ring_merge_size, bs, hs)
                .transpose(0, 2)
                .contiguous()
            )

        return (output, handle) if async_op else output

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# NOTE: this function doesn't support ring attn sharding
def uneven_all_to_all_4D(
    input: Union[torch.tensor, torch.Size],
    ranks_send: List[int],
    ranks_recv: List[int],
    group_send: dist.ProcessGroup,
    group_recv: dist.ProcessGroup,
    dtype: Optional[torch.dtype] = None,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    async_op: bool = False,
    custom_backend: Optional[str] = "torch",
    output: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    Implements uneven all to all (different send & recv ranks) for 4D tensors.
    """
    # logger.warning(
    #     "This function doesn't support ring attn sharding, it is deprecated, please use uneven_decoupled_all_to_all_4D instead"
    # )
    assert custom_backend in [
        "python",
        "nccl",
        "torch",
    ], f"custom_backend must be one of ['python', 'nccl', 'torch'], got {custom_backend}"
    if custom_backend == "torch":
        all_to_all_impl = dist.all_to_all_single
    elif custom_backend == "python":
        all_to_all_impl = custom_all_to_all_single
    elif custom_backend == "nccl":
        all_to_all_impl = custom_all_to_all_single_nccl

    if scatter_idx == 2 and gather_idx == 1:
        assert isinstance(input, torch.Tensor), "input must be a tensor"
        assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

        rank_send = dist.get_global_rank(group_send, group_send.rank())
        non_attn_world_size = dist.get_world_size(group_send)
        if rank_send in ranks_recv:
            attn_world_size = dist.get_world_size(group_recv)
            assert (
                len(ranks_recv) == attn_world_size
            ), f"ranks_recv must have the same length as group_recv, got {len(ranks_recv)} and {attn_world_size}"
        else:
            attn_world_size = len(ranks_recv)

        assert (
            len(ranks_send) == non_attn_world_size
        ), f"ranks_send must have the same length as group_send, got {len(ranks_send)} and {non_attn_world_size}"

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P_send, hc, hs) output: (bs, seqlen, hc/P_recv, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * non_attn_world_size
        shard_hc = hc // attn_world_size

        input_t = (
            input.reshape(bs, shard_seqlen, attn_world_size, shard_hc, hs).transpose(0, 2).contiguous()
        )  # (attn_world_size, shard_seqlen, bs, shard_hc, hs)

        # tensor version
        # input_split_sizes = torch.empty(non_attn_world_size, dtype=torch.int32, device=f"cuda:{rank_send}")
        # split_portion = input_t.shape[0] // len(ranks_recv)
        # input_split_sizes[ranks_recv] = split_portion

        # output_split_sizes = torch.empty_like(input_split_sizes, device=f"cuda:{rank_recv}")
        # NOTE(@runyu): not sure, this may cause small performance degradation
        # dist.all_to_all_single(output_split_sizes, input_split_sizes, group=group_send)
        # input_split_size_list = input_split_sizes.tolist()
        # output_split_size_list = output_split_sizes.tolist()

        # list version
        split_portion = input_t.shape[0] // len(ranks_recv)
        input_split_size_list = [split_portion if i in ranks_recv else 0 for i in range(len(ranks_send))]
        if rank_send in ranks_recv:
            output_split_size_list = [split_portion] * len(ranks_send)
        else:
            output_split_size_list = [0] * len(ranks_send)

        output_shape = list(input_t.shape)
        output_shape[0] = sum(output_split_size_list)
        if output is not None:
            assert output.shape == output_shape, f"output_tensor shape {output.shape} does not match {output_shape}"
        else:
            output = torch.empty(output_shape, device=input_t.device, dtype=input_t.dtype)

        handle = all_to_all_impl(
            output,
            input_t,
            output_split_sizes=output_split_size_list,
            input_split_sizes=input_split_size_list,
            group=group_send,
            async_op=async_op,
        )
        # dist.barrier(group=group_send)

        if not async_op:
            if output_shape[0] != 0:
                output = output.reshape(seqlen, bs, shard_hc, hs)
                output = output.transpose(0, 1).contiguous()
            else:
                output = None

        return (output, handle) if async_op else output

    elif scatter_idx == 1 and gather_idx == 2:  # attn back to mlp

        rank_recv = dist.get_global_rank(group_recv, group_recv.rank())
        non_attn_world_size = dist.get_world_size(group_recv)
        assert (
            len(ranks_recv) == non_attn_world_size
        ), f"ranks_recv must have the same length as group_recv, got {len(ranks_recv)} and {non_attn_world_size}"

        attn_world_size = len(ranks_send)

        if isinstance(input, torch.Tensor):
            # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P_attn, hs) output: (bs, seqlen/P_mlp, hc, hs)
            bs, seqlen, shard_hc, hs = input.shape
            hc = shard_hc * attn_world_size
            shard_seqlen = seqlen // non_attn_world_size

            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            # (bs, seqlen, hc/P_attn, hs) -reshape-> (bs, P_mlp, seq_len/P_mlp, hc/P_attn, hs)
            # -> (P_mlp, hc/P_attn, seqlen/P_mlp, bs, hs)
            input_t = (
                input.reshape(bs, non_attn_world_size, shard_seqlen, shard_hc, hs).permute(1, 3, 2, 0, 4).contiguous()
            )
        elif isinstance(input, torch.Size):
            assert dtype is not None, "When input is empty, receiving dtype must be specified to avoid deadlock."
            bs, seqlen, shard_hc, hs = input
            hc = shard_hc * attn_world_size
            shard_seqlen = seqlen // non_attn_world_size
            # input_t = torch.tensor([], device=f"cuda:{rank_recv}")
            input_t = torch.empty(
                (0, non_attn_world_size, shard_seqlen, shard_hc, hs), device=f"cuda:{rank_recv}", dtype=dtype
            )
        else:
            raise RuntimeError("input must be a tensor or a torch.Size")

        # # tensor version
        # if rank_recv in ranks_send:
        #     input_split_sizes = torch.ones(non_attn_world_size, dtype=torch.int32, device=f"cuda:{rank_recv}")
        # else:
        #     input_split_sizes = torch.empty(non_attn_world_size, dtype=torch.int32, device=f"cuda:{rank_recv}")
        # output_split_sizes = torch.empty_like(input_split_sizes)
        # NOTE(@runyu): not sure, this may cause small performance degradation
        # dist.all_to_all_single(output_split_sizes, input_split_sizes, group=group_recv)
        # input_split_size_list = input_split_sizes.tolist()
        # output_split_size_list = output_split_sizes.tolist()

        # list version
        if rank_recv in ranks_send:
            input_split_size_list = [1] * len(ranks_recv)
        else:
            input_split_size_list = [0] * len(ranks_recv)
        output_split_size_list = [1 if i in ranks_send else 0 for i in range(len(ranks_recv))]
        output_shape = [non_attn_world_size, shard_hc, shard_seqlen, bs, hs]
        output_shape[0] = sum(output_split_size_list)
        # logger.info(
        #     f"rank {dist.get_rank()} input_split_size_list {input_split_size_list}, output_split_size_list {output_split_size_list},"
        #     + f"input_shape {input_t.shape}, output_shape {output_shape}"
        # )
        if output is not None:
            assert output.shape == output_shape, f"output_tensor shape {output.shape} does not match {output_shape}"
        else:
            output = torch.empty(output_shape, device=input_t.device, dtype=input_t.dtype)
        handle = all_to_all_impl(
            output,
            input_t,
            output_split_sizes=output_split_size_list,
            input_split_sizes=input_split_size_list,
            group=group_recv,
            async_op=async_op,
        )
        # dist.barrier(group=group_recv)
        if not async_op:
            output = output.reshape(hc, shard_seqlen, bs, hs)
            output = output.transpose(0, 2).contiguous()

        return (output, handle) if async_op else output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
    async_op: bool = False,
    output: Optional[torch.tensor] = None,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor):  sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        if seq_world_size == 1:
            return input
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t) if output is None else output
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            handle = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(seqlen, bs, shard_hc, hs)

            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return (output, handle) if async_op else output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs).transpose(0, 3).transpose(0, 1).contiguous()
        )

        output = torch.empty_like(input_t) if output is None else output
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            handle = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
            # if use_sync:
            #     torch.cuda.synchronize()
        else:
            output = input_t

        if not async_op:
            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = output.transpose(0, 2).contiguous()

        return (output, handle) if async_op else output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def custom_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    backend: Optional[str] = "python",
) -> Optional[dist.Work]:
    """
    Custom all_to_all_single implementation to avoid deadlock when using PyTorch's implemention
    and some ranks don't send anything.

    Args:
        output (torch.Tensor): The output tensor. Shape should be [sum(output_split_sizes), ...].
        input (torch.Tensor): The input tensor. Shape should be [sum(input_split_sizes), ...].
        output_split_sizes (List[int]): The sizes of the splits for the output tensor.
        input_split_sizes (List[int]): The sizes of the splits for the input tensor.
        group (dist.ProcessGroup): The process group to use for communication.
        async_op (bool, optional): Whether to perform the operation asynchronously. Defaults to False.

    Returns:
        Optional[dist.Work]: The wait handles if async_op is True, otherwise None.
    """
    if backend == "python":
        return custom_all_to_all_single_python(
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            group,
            async_op,
        )
    elif backend == "nccl":
        return custom_all_to_all_single_nccl(
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            group,
            async_op,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def custom_all_to_all_single_python(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # Iterate over the ranks in the process group
    batch_ops = []
    handles = []
    sum_sent = 0
    sum_recv = 0
    num_ops = 0
    for i in range(world_size):
        send_split_size = input_split_sizes[i]
        recv_split_size = output_split_sizes[i]

        if i == rank:
            if send_split_size > 0:
                assert (
                    send_split_size == recv_split_size
                ), f"input_split_sizes {input_split_sizes}, output_split_sizes {output_split_sizes} are not equal"
                output[sum_sent : sum_sent + send_split_size].copy_(input[sum_recv : sum_recv + send_split_size])
                sum_sent += send_split_size
                sum_recv += send_split_size
            continue

        send_rank = dist.get_global_rank(group, i)
        recv_rank = send_rank

        # Alternate order to avoid deadlock
        if rank % 2 == 0:
            if send_split_size > 0:
                send_op = dist.P2POp(
                    dist.isend, input[sum_sent : sum_sent + send_split_size], send_rank, dist.group.WORLD
                )
                batch_ops.append(send_op)
                # handle = dist.irecv(input[sum_sent:sum_sent + send_split_size], send_rank, group)
                # handles.append(handle)
                sum_sent += send_split_size

            if recv_split_size > 0:
                recv_op = dist.P2POp(
                    dist.irecv, output[sum_recv : sum_recv + recv_split_size], recv_rank, dist.group.WORLD
                )
                batch_ops.append(recv_op)
                # handle = dist.isend(output[sum_recv:sum_recv + recv_split_size], recv_rank, group)
                # handles.append(handle)
                sum_recv += recv_split_size
        else:
            if recv_split_size > 0:
                recv_op = dist.P2POp(
                    dist.irecv, output[sum_recv : sum_recv + recv_split_size], recv_rank, dist.group.WORLD
                )
                batch_ops.append(recv_op)
                # handle = dist.isend(output[sum_recv:sum_recv + recv_split_size], recv_rank, group)
                # handles.append(handle)
                sum_recv += recv_split_size

            if send_split_size > 0:
                send_op = dist.P2POp(
                    dist.isend, input[sum_sent : sum_sent + send_split_size], send_rank, dist.group.WORLD
                )
                batch_ops.append(send_op)
                # handle = dist.irecv(input[sum_sent:sum_sent + send_split_size], send_rank, group)
                # handles.append(handle)
                sum_sent += send_split_size

    # Commit batch
    if len(batch_ops) > 0:
        handles = dist.batch_isend_irecv(batch_ops)
    if not async_op:
        for req in handles:
            req.wait()
    else:
        return handles


def custom_all_to_all_single_nccl(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: dist.ProcessGroup,
    async_op: bool = None,
):
    """
    Calls into the c++ implementation of all_to_all_single
    """

    # logger.info(f"rank {get_world_group().rank_in_group} calling nccl AllToAllv!")
    from cfuser.core.distributed.globals import PROCESS_GROUP

    PROCESS_GROUP.custom_nccl_comm.all_to_all_single(
        input, output, input_split_sizes, output_split_sizes, PROCESS_GROUP.custom_nccl_comm.nccl_stream
    )
    # raise NotImplementedError("NCCL backend is not implemented yet.")
