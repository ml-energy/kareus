import os
from typing import Any, Dict, List, Optional

import cupy as cp
import torch
import torch.distributed as dist

from mscclpp_benchmark.mscclpp_op import MscclppAllReduce1, MscclppAllGather
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
    Host2DeviceSemaphore,
    RegisteredMemory,
    MemoryChannel,
    ProxyService,
    TcpBootstrap,
    connect_nvls_collective,
)


# Global state isolated per collective type to allow concurrent groups
# AllReduce state
AR_COMM: Optional[Communicator] = None
AR_COMM_STREAM: Optional[torch.cuda.Stream] = None
_AR_GROUP_RANK: Optional[int] = None
_AR_GROUP_SIZE: Optional[int] = None

_AR_ALGO: Optional[MscclppAllReduce1] = None
_AR_TORCH_OUT_TENSOR: Optional[torch.Tensor] = None
_AR_TORCH_IN_TENSOR: Optional[torch.Tensor] = None
_AR_TORCH_SUBGROUP: Optional[dist.ProcessGroup] = None

# AllGather state
AG_COMM: Optional[Communicator] = None
AG_COMM_STREAM: Optional[torch.cuda.Stream] = None
_AG_GROUP_RANK: Optional[int] = None
_AG_GROUP_SIZE: Optional[int] = None

_AG_ALGO: Optional[Any] = None
_AG_TORCH_OUT_TENSOR: Optional[torch.Tensor] = None
_AG_TORCH_IN_TENSOR: Optional[torch.Tensor] = None
_AG_TORCH_SUBGROUP: Optional[dist.ProcessGroup] = None
_AG_PROXY_SERVICE: Optional[ProxyService] = None
_AG_USE_TORCH_SINGLE_NODE: bool = False


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

    def make_port_channels(self, proxy_service: ProxyService, tensor: Any, connections: Dict[int, Connection]):
        # Create Host2Device semaphores for each connection
        h2d_semaphores: Dict[int, Host2DeviceSemaphore] = {}
        for rank in connections:
            h2d_semaphores[rank] = Host2DeviceSemaphore(self.communicator, connections[rank])

        # Register local tensor with peers and exchange registered memory handles
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
        registered = self._register_memory_with_connections(local_reg, connections)

        # Add memories and semaphores to proxy service
        memory_ids: Dict[int, int] = {}
        semaphore_ids: Dict[int, int] = {}
        for rank in registered:
            memory_ids[rank] = proxy_service.add_memory(registered[rank])
        for rank in h2d_semaphores:
            semaphore_ids[rank] = proxy_service.add_semaphore(h2d_semaphores[rank])

        # Build PortChannels via proxy service
        channels = {}
        for rank in h2d_semaphores:
            channels[rank] = proxy_service.port_channel(semaphore_ids[rank], memory_ids[rank], memory_ids[self.my_rank])
        return channels


def msccl_AllReduce_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
):
    global AR_COMM, AR_COMM_STREAM, _AR_GROUP_RANK, _AR_GROUP_SIZE
    global _AR_ALGO, _AR_TORCH_OUT_TENSOR, _AR_TORCH_IN_TENSOR, _AR_TORCH_SUBGROUP

    if group is not None:
        sub_group = group
        sub_rank = dist.get_rank(group=sub_group)
        sub_world_size = dist.get_world_size(group=sub_group)
    else:
        sub_group = dist.group.WORLD
        sub_rank = rank
        sub_world_size = world_size

    need_reinit = AR_COMM is None or _AR_GROUP_RANK != sub_rank or _AR_GROUP_SIZE != sub_world_size

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

        AR_COMM = Communicator(bootstrap)
        _AR_GROUP_RANK = sub_rank
        _AR_GROUP_SIZE = sub_world_size

    if AR_COMM_STREAM is None:
        AR_COMM_STREAM = torch.cuda.Stream()

    # dtype handling
    # Primary support: fp16/fp32/int32. If bf16 is provided, cast to fp16 for perf-only runs.
    # TODO: support bf16 in mscclpp kernel
    cast_to_fp16 = False
    if input_tensor.dtype == torch.bfloat16:
        cast_to_fp16 = True

    # Build ShimGroup and algo
    _AR_TORCH_SUBGROUP = sub_group
    group_obj = _ShimGroup(AR_COMM, _AR_GROUP_RANK, _AR_GROUP_SIZE, _AR_TORCH_SUBGROUP)

    # Prepare working buffer
    if cast_to_fp16:
        # Create an fp16 working tensor for in-place allreduce; skip copying back for perf
        input_work = input_tensor.to(torch.float16)
    else:
        input_work = input_tensor

    cp_in = _dlpack_view(input_work)

    # MscclppAllReduce1 is in-place; use input buffer as working/output.
    _AR_ALGO = MscclppAllReduce1(group_obj, cp_in)
    # TODO: currently return the original tensor if bf16
    _AR_TORCH_OUT_TENSOR = input_tensor
    _AR_TORCH_IN_TENSOR = input_tensor


def msccl_AllReduce(nblocks: int, block_size: int):
    if _AR_ALGO is None:
        raise RuntimeError("Call msccl_AllReduce_init(...) first")
    # Update launch configuration
    # read_only=0 avoids potential cache staleness on some GPUs
    _AR_ALGO.set_params(nblocks=nblocks, block_size=block_size, read_only=0)
    # Launch on the user-managed torch stream; Kernel launcher supports torch stream with .cuda_stream
    _AR_ALGO(AR_COMM_STREAM)
    return _AR_TORCH_OUT_TENSOR


def msccl_sync():
    if AR_COMM_STREAM is not None:
        AR_COMM_STREAM.synchronize()
    if AG_COMM_STREAM is not None:
        AG_COMM_STREAM.synchronize()


def msccl_AllGather_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    nranks_per_node: int = 0,
):
    global AG_COMM, AG_COMM_STREAM, _AG_GROUP_RANK, _AG_GROUP_SIZE
    global _AG_ALGO, _AG_TORCH_OUT_TENSOR, _AG_TORCH_IN_TENSOR, _AG_TORCH_SUBGROUP, _AG_PROXY_SERVICE

    if group is not None:
        sub_group = group
        sub_rank = dist.get_rank(group=sub_group)
        sub_world_size = dist.get_world_size(group=sub_group)
    else:
        sub_group = dist.group.WORLD
        sub_rank = rank
        sub_world_size = world_size

    need_reinit = AG_COMM is None or _AG_GROUP_RANK != sub_rank or _AG_GROUP_SIZE != sub_world_size

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

        AG_COMM = Communicator(bootstrap)
        _AG_GROUP_RANK = sub_rank
        _AG_GROUP_SIZE = sub_world_size

    if AG_COMM_STREAM is None:
        AG_COMM_STREAM = torch.cuda.Stream()

    # dtype handling
    cast_to_fp16 = False
    if input_tensor.dtype == torch.bfloat16:
        cast_to_fp16 = True

    _AG_TORCH_SUBGROUP = sub_group
    group_obj = _ShimGroup(AG_COMM, _AG_GROUP_RANK, _AG_GROUP_SIZE, _AG_TORCH_SUBGROUP)

    # Working buffer: operate in-place on the provided buffer tensor
    if cast_to_fp16:
        work_buf = input_tensor.to(torch.float16)
    else:
        work_buf = input_tensor

    cp_mem = _dlpack_view(work_buf)

    # Determine single-node vs multi-node
    if nranks_per_node <= 0:
        nranks_per_node = _AG_GROUP_SIZE
    _AG_USE_TORCH_SINGLE_NODE = (nranks_per_node == _AG_GROUP_SIZE)

    if _AG_USE_TORCH_SINGLE_NODE:
        _AG_ALGO = MscclppAllGather(group_obj, cp_mem, nranks_per_node, None)
        _AG_PROXY_SERVICE = None
    else:
        if _AG_PROXY_SERVICE is None:
            _AG_PROXY_SERVICE = ProxyService()
        _AG_ALGO = MscclppAllGather(group_obj, cp_mem, nranks_per_node, _AG_PROXY_SERVICE)
    _AG_TORCH_OUT_TENSOR = input_tensor
    _AG_TORCH_IN_TENSOR = input_tensor


# TODO: pipeline_depth is for multi-node allgather only
def msccl_AllGather(nblocks: int, block_size: int, pipeline_depth: int = 3):
    if _AG_ALGO is None:
        raise RuntimeError("Call msccl_AllGather_init(...) first")
    # Update launch configuration
    _AG_ALGO.set_params(nblocks=nblocks, block_size=block_size, pipeline_depth=pipeline_depth)
    # Ensure proxy service is running
    global _AG_PROXY_SERVICE
    # TODO: start_proxy in init?
    if _AG_PROXY_SERVICE is not None:
        try:
            _AG_PROXY_SERVICE.start_proxy()
        except Exception:
            pass
    # Launch
    _AG_ALGO(AG_COMM_STREAM)
    return _AG_TORCH_OUT_TENSOR


def msccl_cleanup():
    """Release global objects to avoid nanobind leak warnings at exit."""
    global AR_COMM, AR_COMM_STREAM, _AR_GROUP_RANK, _AR_GROUP_SIZE
    global _AR_ALGO, _AR_TORCH_OUT_TENSOR, _AR_TORCH_IN_TENSOR, _AR_TORCH_SUBGROUP
    global AG_COMM, AG_COMM_STREAM, _AG_GROUP_RANK, _AG_GROUP_SIZE
    global _AG_ALGO, _AG_TORCH_OUT_TENSOR, _AG_TORCH_IN_TENSOR, _AG_TORCH_SUBGROUP, _AG_PROXY_SERVICE
    try:
        # Ensure all work is done before tearing down
        if dist.is_initialized():
            try:
                dist.barrier(group=_AR_TORCH_SUBGROUP if _AR_TORCH_SUBGROUP is not None else dist.group.WORLD)
            except Exception:
                pass
            try:
                dist.barrier(group=_AG_TORCH_SUBGROUP if _AG_TORCH_SUBGROUP is not None else dist.group.WORLD)
            except Exception:
                pass
        torch.cuda.synchronize()
    except Exception:
        pass

    _AR_ALGO = None
    AR_COMM = None
    AR_COMM_STREAM = None
    _AR_TORCH_OUT_TENSOR = None
    _AR_TORCH_IN_TENSOR = None
    _AR_TORCH_SUBGROUP = None
    _AR_GROUP_RANK = None
    _AR_GROUP_SIZE = None

    _AG_ALGO = None
    AG_COMM = None
    AG_COMM_STREAM = None
    _AG_TORCH_OUT_TENSOR = None
    _AG_TORCH_IN_TENSOR = None
    _AG_TORCH_SUBGROUP = None
    _AG_GROUP_RANK = None
    _AG_GROUP_SIZE = None

    # Stop and clear ProxyService
    try:
        if _AG_PROXY_SERVICE is not None:
            _AG_PROXY_SERVICE.stop_proxy()
    except Exception:
        pass
    _AG_PROXY_SERVICE = None

    # Free CuPy memory pools to drop device allocations
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

