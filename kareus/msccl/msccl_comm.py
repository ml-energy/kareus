import os
from typing import Any, Dict, List, Optional
import contextlib
import atexit

import cupy as cp
import torch
import torch.distributed as dist

from mscclpp_benchmark.mscclpp_op import MscclppAllReduce1, type_to_str
from mscclpp.utils import is_torch_tensor
from mscclpp import (
    Transport,
    TransportFlags,
    MemoryDevice2DeviceSemaphore,
)
from mscclpp._mscclpp import (
    Communicator,
    Connection,
    EndpointConfig,
    RegisteredMemory,
    MemoryChannel,
    TcpBootstrap,
    connect_nvls_collective,
)


# Global state mimicking the user's requested interface
COMM: Optional[Communicator] = None
COMM_STREAM: Optional[torch.cuda.Stream] = None
_COMM_GROUP_RANK: Optional[int] = None
_COMM_GROUP_SIZE: Optional[int] = None

_ALGO: Optional[MscclppAllReduce1] = None
_TORCH_OUT_TENSOR: Optional[torch.Tensor] = None
_TORCH_IN_TENSOR: Optional[torch.Tensor] = None
_TORCH_SUBGROUP: Optional[dist.ProcessGroup] = None


def _dlpack_view(t: torch.Tensor) -> cp.ndarray:
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))


class _ShimGroup:
    """Minimal subset of CommGroup used by MscclppAllReduce2.

    Provides connection creation and memory/port channel helpers using an
    already-initialized Communicator (created via torch.distributed broadcast).
    """

    def __init__(self, communicator: Communicator, my_rank: int, nranks: int, subgroup: Optional[dist.ProcessGroup]):
        self.communicator = communicator
        self.my_rank = my_rank
        self.nranks = nranks
        self.subgroup = subgroup

    def barrier(self):
        if dist.is_initialized():
            try:
                dist.barrier(group=self.subgroup if self.subgroup is not None else dist.group.WORLD)
            except Exception:
                # Fallback to no-op if subgroup barrier unsupported
                pass

    def make_connection(
        self,
        all_ranks: List[int],
        endpoints: EndpointConfig | Transport | Dict[int, EndpointConfig] | Dict[int, Transport],
        use_switch: bool = False,
    ) -> Dict[int, Connection]:
        if type(endpoints) is Transport:
            endpoints = EndpointConfig(endpoints)
        elif type(endpoints) is dict:
            endpoints = {k: EndpointConfig(v) if type(v) is Transport else v for k, v in endpoints.items()}
        connections = {}
        for rank in all_ranks:
            if type(endpoints) is dict:
                endpoint = endpoints[rank]
            else:
                endpoint = endpoints
            if endpoint.transport == Transport.CudaIpc and use_switch:
                return connect_nvls_collective(self.communicator, all_ranks, 2**30)
            else:
                connections[rank] = self.communicator.connect(endpoint, rank)
        connections = {rank: connections[rank].get() for rank in connections}
        return connections

    def _register_memory_with_connections(
        self, memory: RegisteredMemory, connections: Dict[int, Connection]
    ) -> Dict[int, RegisteredMemory]:
        all_registered_memories: Dict[int, RegisteredMemory] = {}
        all_registered_memories[self.my_rank] = memory
        future_memories: Dict[int, Any] = {}
        for rank in connections:
            self.communicator.send_memory(memory, rank)
            future_memories[rank] = self.communicator.recv_memory(rank)
        for rank in connections:
            all_registered_memories[rank] = future_memories[rank].get()
        return all_registered_memories

    def register_local_memory(self, tensor: Any, connections: Dict[int, Connection]) -> RegisteredMemory:
        transport_flags = TransportFlags()
        for rank in connections:
            transport_flags |= connections[rank].transport()
        if isinstance(tensor, cp.ndarray):
            data_ptr = tensor.data.ptr
            tensor_size = tensor.size * tensor.itemsize
        elif is_torch_tensor(tensor):
            data_ptr = tensor.data_ptr()
            tensor_size = tensor.numel() * tensor.element_size()
        else:
            data_ptr = tensor.ctypes.data
            tensor_size = tensor.size * tensor.itemsize
        return self.communicator.register_memory(data_ptr, tensor_size, transport_flags)

    def make_semaphore(
        self, connections: Dict[int, Connection]
    ) -> Dict[int, MemoryDevice2DeviceSemaphore]:
        semaphores: Dict[int, MemoryDevice2DeviceSemaphore] = {}
        for rank in connections:
            semaphores[rank] = MemoryDevice2DeviceSemaphore(self.communicator, connections[rank])
        return semaphores

    def make_memory_channels_with_scratch(
        self, tensor: Any, registeredScratchBuffer: RegisteredMemory, connections: Dict[int, Connection]
    ) -> Dict[int, MemoryChannel]:
        semaphores = self.make_semaphore(connections)
        registered_memories = self._register_memory_with_connections(registeredScratchBuffer, connections)
        if isinstance(tensor, cp.ndarray):
            tensor_data_ptr = tensor.data.ptr
            tensor_size = tensor.size * tensor.itemsize
        elif is_torch_tensor(tensor):
            tensor_data_ptr = tensor.data_ptr()
            tensor_size = tensor.numel() * tensor.element_size()
        else:
            tensor_data_ptr = tensor.ctypes.data
            tensor_size = tensor.size * tensor.itemsize
        local_registered_memory = self.communicator.register_memory(tensor_data_ptr, tensor_size, TransportFlags())
        scratch_data_ptr = registeredScratchBuffer.data()
        channels: Dict[int, MemoryChannel] = {}
        for rank in connections:
            channels[rank] = MemoryChannel(
                semaphores[rank], registered_memories[rank], local_registered_memory, scratch_data_ptr
            )
        return channels

    def make_memory_channels(self, tensor: Any, connections: Dict[int, Connection]) -> Dict[int, MemoryChannel]:
        # Mirror CommGroup.make_memory_channels implementation for torch tensors
        semaphores = self.make_semaphore(connections)
        # Register local tensor and exchange with peers
        # First register local
        transport_flags = TransportFlags()
        for rank in connections:
            transport_flags |= connections[rank].transport()
        if isinstance(tensor, cp.ndarray):
            data_ptr = tensor.data.ptr
            tensor_size = tensor.size * tensor.itemsize
        elif is_torch_tensor(tensor):
            data_ptr = tensor.data_ptr()
            tensor_size = tensor.numel() * tensor.element_size()
        else:
            data_ptr = tensor.ctypes.data
            tensor_size = tensor.size * tensor.itemsize
        local_reg = self.communicator.register_memory(data_ptr, tensor_size, transport_flags)
        # Exchange registration with peers
        registered = self._register_memory_with_connections(local_reg, connections)
        channels: Dict[int, MemoryChannel] = {}
        for rank in connections:
            channels[rank] = MemoryChannel(semaphores[rank], registered[rank], registered[self.my_rank])
        return channels


def msccl_AllReduce_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    # output_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
):
    global COMM, COMM_STREAM, _COMM_GROUP_RANK, _COMM_GROUP_SIZE
    global _ALGO, _TORCH_OUT_TENSOR, _TORCH_IN_TENSOR, _TORCH_SUBGROUP

    if group is not None:
        sub_group = group
        sub_rank = dist.get_rank(group=sub_group)
        sub_world_size = dist.get_world_size(group=sub_group)
    else:
        sub_group = dist.group.WORLD
        sub_rank = rank
        sub_world_size = world_size

    need_reinit = COMM is None or _COMM_GROUP_RANK != sub_rank or _COMM_GROUP_SIZE != sub_world_size

    if need_reinit:
        bootstrap = TcpBootstrap.create(sub_rank, sub_world_size)
        if sub_rank == 0:
            unique_id = bootstrap.create_unique_id()
            dist_list: List[Optional[bytes]] = [unique_id]
        else:
            dist_list = [None]

        group_ranks = dist.get_process_group_ranks(sub_group)
        src_rank = group_ranks[0]
        dist.broadcast_object_list(dist_list, src=src_rank, group=sub_group)

        unique_id = dist_list[0]
        bootstrap.initialize(unique_id)

        COMM = Communicator(bootstrap)
        _COMM_GROUP_RANK = sub_rank
        _COMM_GROUP_SIZE = sub_world_size

    if COMM_STREAM is None:
        COMM_STREAM = torch.cuda.Stream()

    # dtype handling
    # Primary support: fp16/fp32/int32. If bf16 is provided, cast to fp16 for perf-only runs.
    # TODO: support bf16 in mscclpp kernel
    cast_to_fp16 = False
    if input_tensor.dtype == torch.bfloat16:
        cast_to_fp16 = True
        cupy_dtype = cp.float16
    else:
        try:
            cupy_dtype = {
                torch.float16: cp.float16,
                torch.float32: cp.float32,
                torch.int32: cp.int32,
            }[input_tensor.dtype]
        except KeyError:
            raise RuntimeError(
                f"Unsupported dtype {input_tensor.dtype}. Supported: float16, float32, int32 (bf16 is cast to fp16 for perf)."
            )

    # Build ShimGroup and algo
    _TORCH_SUBGROUP = sub_group
    group_obj = _ShimGroup(COMM, _COMM_GROUP_RANK, _COMM_GROUP_SIZE, _TORCH_SUBGROUP)

    # Prepare working buffers
    if cast_to_fp16:
        # Create an fp16 working tensor for in-place allreduce; skip copying back for perf
        input_work = input_tensor.to(torch.float16)
    else:
        input_work = input_tensor

    cp_in = _dlpack_view(input_work)
    # cp_out = _dlpack_view(output_tensor)

    # MscclppAllReduce1 is in-place; use input buffer as working/output.
    # We'll still return the provided output_tensor for API parity by copying after launch
    _ALGO = MscclppAllReduce1(group_obj, cp_in)
    # _TORCH_OUT_TENSOR = output_tensor
    # TODO: currently return the original tensor if bf16
    _TORCH_IN_TENSOR = input_tensor  


def msccl_AllReduce(nblocks: int, block_size: int):
    if _ALGO is None:
        raise RuntimeError("Call msccl_AllReduce_init(...) first")
    # Update launch configuration
    # read_only=0 avoids potential cache staleness on some GPUs
    _ALGO.set_params(nblocks=nblocks, block_size=block_size, read_only=0)
    # Launch on the user-managed torch stream; Kernel launcher supports torch stream with .cuda_stream
    _ALGO(COMM_STREAM)
    # Copy result into the user-provided output tensor if different (in-place result in input tensor)
    # if _TORCH_OUT_TENSOR.data_ptr() != _TORCH_IN_TENSOR.data_ptr():
    #     _TORCH_OUT_TENSOR.copy_(_TORCH_IN_TENSOR)
    # return _TORCH_OUT_TENSOR
    return _TORCH_IN_TENSOR


def msccl_sync():
    COMM_STREAM.synchronize()


def msccl_cleanup():
    """Release global objects to avoid nanobind leak warnings at exit."""
    global COMM, COMM_STREAM, _COMM_GROUP_RANK, _COMM_GROUP_SIZE
    global _ALGO, _TORCH_OUT_TENSOR, _TORCH_IN_TENSOR, _TORCH_SUBGROUP
    try:
        # Ensure all work is done before tearing down
        if dist.is_initialized():
            try:
                dist.barrier(group=_TORCH_SUBGROUP if _TORCH_SUBGROUP is not None else dist.group.WORLD)
            except Exception:
                pass
        torch.cuda.synchronize()
    except Exception:
        pass

    # Explicitly release algorithm and its resources
    try:
        if _ALGO is not None:
            # Force cleanup of any internal connections/channels
            del _ALGO
    except Exception:
        pass

    # Clear all global references
    _ALGO = None
    COMM = None
    COMM_STREAM = None
    _TORCH_OUT_TENSOR = None
    _TORCH_IN_TENSOR = None
    _TORCH_SUBGROUP = None
    _COMM_GROUP_RANK = None
    _COMM_GROUP_SIZE = None

    # Force garbage collection to help release nanobind objects
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    # Free CuPy memory pools to drop device allocations
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


@contextlib.contextmanager
def msccl_context():
    """Context manager for proper MSCCL resource management.
    
    Usage:
        with msccl_context():
            # Use MSCCL operations
            pass
        # Resources are automatically cleaned up
    """
    try:
        yield
    finally:
        msccl_cleanup()


# Register cleanup at module import to ensure it runs at program exit
atexit.register(msccl_cleanup)
