import os
from typing import Any, Dict, List, Optional

import cupy as cp
import torch
import torch.distributed as dist

from mscclpp_benchmark.mscclpp_op import MscclppAllReduce1, MscclppAllGather, MscclppReduceScatter
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

# TODO: create different op each request
# TODO: ProxyService start in init? multi-node support


def _dlpack_view(t: torch.Tensor) -> cp.ndarray:
    """Convert torch tensor to CuPy array via DLPack.
    
    For bfloat16 tensors, we view them as uint16 since CuPy doesn't have
    native bfloat16 support. The CUDA kernels will interpret the data correctly.
    """
    if t.dtype == torch.bfloat16:
        # View as uint16 for CuPy (both are 16-bit)
        t_view = t.view(dtype=torch.uint16)
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t_view))
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


class AllReduceManager:
    """Manages AllReduce communicator, stream, and kernel state."""

    def __init__(self) -> None:
        self.communicator: Optional[Communicator] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.group_rank: Optional[int] = None
        self.group_size: Optional[int] = None
        self.subgroup: Optional[dist.ProcessGroup] = None
        self.algo: Optional[MscclppAllReduce1] = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.input_tensor: Optional[torch.Tensor] = None

    def init(
        self,
        rank: int,
        world_size: int,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
    ) -> None:
        if group is not None:
            sub_group = group
            sub_rank = dist.get_rank(group=sub_group)
            sub_world_size = dist.get_world_size(group=sub_group)
        else:
            sub_group = dist.group.WORLD
            sub_rank = rank
            sub_world_size = world_size

        need_reinit = (
            self.communicator is None
            or self.group_rank != sub_rank
            or self.group_size != sub_world_size
        )

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

            self.communicator = Communicator(bootstrap)
            self.group_rank = sub_rank
            self.group_size = sub_world_size

        global SHARED_COMM_STREAM
        if SHARED_COMM_STREAM is None:
            SHARED_COMM_STREAM = torch.cuda.Stream()
        self.stream = SHARED_COMM_STREAM

        # cast_to_fp16 = False
        # if input_tensor.dtype == torch.bfloat16:
        #     cast_to_fp16 = True

        self.subgroup = sub_group
        if need_reinit and self.algo is None:
            group_obj = _ShimGroup(self.communicator, self.group_rank, self.group_size, self.subgroup)  # type: ignore[arg-type]

            # new_size = list(input_tensor.size())
            # input_work = input_tensor.to(torch.float16)
            input_work = input_tensor

            cp_in = _dlpack_view(input_work)

            self.algo = MscclppAllReduce1(group_obj, cp_in)
            self.output_tensor = input_tensor
            self.input_tensor = input_tensor

    def __call__(self, nblocks: int, block_size: int) -> torch.Tensor:
        if self.algo is None or self.stream is None or self.output_tensor is None:
            raise RuntimeError("Call init(...) before launching AllReduce")
        self.algo.set_params(nblocks=nblocks, block_size=block_size, read_only=0)
        self.algo(self.stream)
        return self.output_tensor

    def sync(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()

    def cleanup(self) -> None:
        try:
            if dist.is_initialized():
                try:
                    dist.barrier(group=self.subgroup if self.subgroup is not None else dist.group.WORLD)
                except Exception:
                    pass
            torch.cuda.synchronize()
        except Exception:
            pass
        self.algo = None
        self.communicator = None
        self.stream = None
        self.output_tensor = None
        self.input_tensor = None
        self.subgroup = None
        self.group_rank = None
        self.group_size = None


class AllGatherManager:
    """Manages AllGather communicator, stream, proxy service, and kernel state."""

    def __init__(self) -> None:
        self.communicator: Optional[Communicator] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.group_rank: Optional[int] = None
        self.group_size: Optional[int] = None
        self.subgroup: Optional[dist.ProcessGroup] = None
        self.algo: Optional[Any] = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.input_tensor: Optional[torch.Tensor] = None
        self.proxy_service: Optional[ProxyService] = None
        self.use_torch_single_node: bool = False

    def init(
        self,
        rank: int,
        world_size: int,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
        nranks_per_node: int = 0,
    ) -> None:
        if group is not None:
            sub_group = group
            sub_rank = dist.get_rank(group=sub_group)
            sub_world_size = dist.get_world_size(group=sub_group)
        else:
            sub_group = dist.group.WORLD
            sub_rank = rank
            sub_world_size = world_size

        need_reinit = (
            self.communicator is None
            or self.group_rank != sub_rank
            or self.group_size != sub_world_size
        )

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

            self.communicator = Communicator(bootstrap)
            self.group_rank = sub_rank
            self.group_size = sub_world_size

        global SHARED_COMM_STREAM
        if SHARED_COMM_STREAM is None:
            SHARED_COMM_STREAM = torch.cuda.Stream()
        self.stream = SHARED_COMM_STREAM

        # cast_to_fp16 = False
        # if input_tensor.dtype == torch.bfloat16:
        #     cast_to_fp16 = True

        self.subgroup = sub_group
        group_obj = _ShimGroup(self.communicator, self.group_rank, self.group_size, self.subgroup)  # type: ignore[arg-type]

        # if cast_to_fp16:
        #     work_buf1 = input_tensor.to(torch.float16)
        #     work_buf2 = input_tensor.to(torch.float16)
        # else:
        #     work_buf1 = input_tensor.clone()
        #     work_buf2 = input_tensor.clone()
        work_buf1 = input_tensor
        work_buf2 = input_tensor

        cp_mem1 = _dlpack_view(work_buf1)
        cp_mem2 = _dlpack_view(work_buf2)

        if nranks_per_node <= 0:
            nranks_per_node = self.group_size if self.group_size is not None else 0
        self.use_torch_single_node = (nranks_per_node == self.group_size)

        if self.use_torch_single_node:
            self.algo = MscclppAllGather(group_obj, cp_mem1, cp_mem2, nranks_per_node, None)
            self.proxy_service = None
        else:
            if self.proxy_service is None:
                self.proxy_service = ProxyService()
            self.algo = MscclppAllGather(group_obj, cp_mem1, cp_mem2, nranks_per_node, self.proxy_service)

        self.output_tensor = input_tensor
        self.input_tensor = input_tensor

    def __call__(self, nblocks: int, block_size: int, pipeline_depth: int = 3) -> torch.Tensor:
        if self.algo is None or self.stream is None or self.output_tensor is None:
            raise RuntimeError("Call init(...) before launching AllGather")
        self.algo.set_params(nblocks=nblocks, block_size=block_size, pipeline_depth=pipeline_depth)
        if self.proxy_service is not None:
            try:
                self.proxy_service.start_proxy()
            except Exception:
                pass
        self.algo(self.stream)
        return self.output_tensor

    def sync(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()

    def cleanup(self) -> None:
        try:
            if dist.is_initialized():
                try:
                    dist.barrier(group=self.subgroup if self.subgroup is not None else dist.group.WORLD)
                except Exception:
                    pass
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            if self.proxy_service is not None:
                self.proxy_service.stop_proxy()
        except Exception:
            pass
        self.algo = None
        self.communicator = None
        self.stream = None
        self.output_tensor = None
        self.input_tensor = None
        self.subgroup = None
        self.group_rank = None
        self.group_size = None
        self.proxy_service = None
        self.use_torch_single_node = False


class ReduceScatterManager:
    """Manages ReduceScatter communicator, stream, proxy service, and kernel state."""

    def __init__(self) -> None:
        self.communicator: Optional[Communicator] = None
        self.stream: Optional[torch.cuda.Stream] = None
        self.group_rank: Optional[int] = None
        self.group_size: Optional[int] = None
        self.subgroup: Optional[dist.ProcessGroup] = None
        self.algo: Optional[Any] = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.input_tensor: Optional[torch.Tensor] = None
        self.proxy_service: Optional[ProxyService] = None
        self.use_torch_single_node: bool = False

    def init(
        self,
        rank: int,
        world_size: int,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup = None,
        nranks_per_node: int = 0,
    ) -> None:
        if group is not None:
            sub_group = group
            sub_rank = dist.get_rank(group=sub_group)
            sub_world_size = dist.get_world_size(group=sub_group)
        else:
            sub_group = dist.group.WORLD
            sub_rank = rank
            sub_world_size = world_size

        need_reinit = (
            self.communicator is None
            or self.group_rank != sub_rank
            or self.group_size != sub_world_size
        )

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

            self.communicator = Communicator(bootstrap)
            self.group_rank = sub_rank
            self.group_size = sub_world_size

        global SHARED_COMM_STREAM
        if SHARED_COMM_STREAM is None:
            SHARED_COMM_STREAM = torch.cuda.Stream()
        self.stream = SHARED_COMM_STREAM

        # cast_to_fp16 = False
        # if input_tensor.dtype == torch.bfloat16:
        #     cast_to_fp16 = True

        self.subgroup = sub_group
        if need_reinit and self.algo is None:
            group_obj = _ShimGroup(self.communicator, self.group_rank, self.group_size, self.subgroup)  # type: ignore[arg-type]

            # new_size = list(input_tensor.size())
            # work_buf1 = input_tensor.to(torch.float16)
            # work_buf2 = input_tensor.to(torch.float16)
            work_buf1 = input_tensor
            work_buf2 = input_tensor

            cp_mem1 = _dlpack_view(work_buf1)
            cp_mem2 = _dlpack_view(work_buf2)

            if nranks_per_node <= 0:
                nranks_per_node = self.group_size if self.group_size is not None else 0
            self.use_torch_single_node = (nranks_per_node == self.group_size)

            if self.use_torch_single_node:
                self.algo = MscclppReduceScatter(group_obj, cp_mem1, cp_mem2, nranks_per_node, None)
                self.proxy_service = None
            else:
                if self.proxy_service is None:
                    self.proxy_service = ProxyService()
                self.algo = MscclppReduceScatter(group_obj, cp_mem1, cp_mem2, nranks_per_node, self.proxy_service)

            self.output_tensor = input_tensor
            self.input_tensor = input_tensor

    def __call__(self, nblocks: int, block_size: int, pipeline_depth: int = 3) -> torch.Tensor:
        if self.algo is None or self.stream is None or self.output_tensor is None:
            raise RuntimeError("Call init(...) before launching ReduceScatter")
        self.algo.set_params(nblocks=nblocks, block_size=block_size, pipeline_depth=pipeline_depth)
        if self.proxy_service is not None:
            try:
                self.proxy_service.start_proxy()
            except Exception:
                pass
        self.algo(self.stream)
        return self.output_tensor

    def sync(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()

    def cleanup(self) -> None:
        try:
            if dist.is_initialized():
                try:
                    dist.barrier(group=self.subgroup if self.subgroup is not None else dist.group.WORLD)
                except Exception:
                    pass
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            if self.proxy_service is not None:
                self.proxy_service.stop_proxy()
        except Exception:
            pass
        self.algo = None
        self.communicator = None
        self.stream = None
        self.output_tensor = None
        self.input_tensor = None
        self.subgroup = None
        self.group_rank = None
        self.group_size = None
        self.proxy_service = None
        self.use_torch_single_node = False


# Module-level singletons
_AR_MANAGER: AllReduceManager = AllReduceManager()
_AG_MANAGER: AllGatherManager = AllGatherManager()
_RS_MANAGER: ReduceScatterManager = ReduceScatterManager()


# Shared communication stream for all collectives
SHARED_COMM_STREAM: Optional[torch.cuda.Stream] = None

# Exposed streams for compatibility with callers expecting module attributes
AR_COMM_STREAM: Optional[torch.cuda.Stream] = None
AG_COMM_STREAM: Optional[torch.cuda.Stream] = None
RS_COMM_STREAM: Optional[torch.cuda.Stream] = None


def msccl_AllReduce_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
):
    _AR_MANAGER.init(rank, world_size, input_tensor, group)
    global AR_COMM_STREAM
    AR_COMM_STREAM = _AR_MANAGER.stream


def msccl_AllReduce(nblocks: int, block_size: int):
    return _AR_MANAGER(nblocks, block_size)


def msccl_AllReduce_sync():
    _AR_MANAGER.sync()


def msccl_AllGather_sync():
    _AG_MANAGER.sync()


def msccl_AllGather_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    nranks_per_node: int = 0,
):
    _AG_MANAGER.init(rank, world_size, input_tensor, group, nranks_per_node)
    global AG_COMM_STREAM
    AG_COMM_STREAM = _AG_MANAGER.stream


def msccl_AllGather(nblocks: int, block_size: int, pipeline_depth: int = 3):
    return _AG_MANAGER(nblocks, block_size, pipeline_depth)


def msccl_ReduceScatter_init(
    rank: int,
    world_size: int,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    nranks_per_node: int = 0,
):
    _RS_MANAGER.init(rank, world_size, input_tensor, group, nranks_per_node)
    global RS_COMM_STREAM
    RS_COMM_STREAM = _RS_MANAGER.stream


def msccl_ReduceScatter(nblocks: int, block_size: int, pipeline_depth: int = 3):
    return _RS_MANAGER(nblocks, block_size, pipeline_depth)


def msccl_ReduceScatter_sync():
    _RS_MANAGER.sync()


def msccl_cleanup():
    """Release manager state to avoid nanobind leak warnings at exit."""
    _AR_MANAGER.cleanup()
    _AG_MANAGER.cleanup()
    _RS_MANAGER.cleanup()
    # Clear exposed streams
    global AR_COMM_STREAM, AG_COMM_STREAM, RS_COMM_STREAM, SHARED_COMM_STREAM
    AR_COMM_STREAM = None
    AG_COMM_STREAM = None
    RS_COMM_STREAM = None
    SHARED_COMM_STREAM = None
    # Free CuPy memory pools to drop device allocations
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

