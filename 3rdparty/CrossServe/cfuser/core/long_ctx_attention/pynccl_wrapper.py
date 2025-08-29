# Modified from vllm to use alltoallv
# (https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py#L127)
# SPDX-License-Identifier: Apache-2.0

# This file is a pure Python wrapper for the NCCL library.
# The main purpose is to use NCCL combined with CUDA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls NCCL correctly, but `cupy` itself
#  often gets stuck when initializing the NCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential cuda APIs, that are not allowed during
#  capturing the CUDA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for NCCL. It is usually
# doable, but we often encounter issues related with nccl versions, and need
# to switch between different versions of NCCL. See
# https://github.com/NVIDIA/nccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of NCCL by
# changing the environment variable `VLLM_NCCL_SO_PATH`, or the `so_file`
# variable in the code.

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ReduceOp

from torch.cuda import current_stream
from torch.distributed import ProcessGroup
import torch.distributed as dist

from .utils import computeLengthsAndOffsets


# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class PyNcclCommunicator:
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        # Hardcode path in default dockerfile container.
        # Use `find / -name "libnccl.so*"` to find the path.
        library_path: Optional[str] = "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        custom_nccl_library_path: Optional[str] = None,  # such as csrc/comm/build/libcustom_nccl_all2all.so
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """

        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group = group
        self.nccl_stream = torch.cuda.Stream(device)

        assert self.world_size >= 1, "No need to create a communicator for world_size 1"
        self.nccl = NCCLLibrary(library_path, custom_nccl_library_path)
        self.available = True
        self.disabled = False

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
            broadcast_list = [self.unique_id]
        else:
            broadcast_list = [None]
        dist.broadcast_object_list(broadcast_list, src=0, group=group)

        if self.rank != 0:
            self.unique_id = broadcast_list[0]

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(self.world_size, self.unique_id, self.rank)

    def all_reduce(self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None) -> torch.Tensor:

        if stream is None:
            stream = self.nccl_stream
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        out_tensor = torch.empty_like(in_tensor)
        self.nccl.ncclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_to_all(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        stream=None,
    ) -> None:
        if stream is None:
            stream = self.nccl_stream
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        count = in_tensor[0].numel()

        out_tensor = torch.empty_like(in_tensor) if out_tensor is None else out_tensor

        self.nccl.ncclAllToAll(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            count,
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        # stream.synchronize() # sync the stream to make sure the all_to_all is finished
        return out_tensor

    def all_to_all_single(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor,
        input_split_sizes: List[int],
        output_split_sizes: List[int],
        stream=None,
    ) -> None:
        if stream is None:
            stream = self.nccl_stream
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )
        assert out_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the output tensor is on {out_tensor.device}"
        )
        if in_tensor.numel() == 0:
            in_tensor = torch.empty([1], device=in_tensor.device, dtype=in_tensor.dtype)
        input_lengths, input_offsets = computeLengthsAndOffsets(input_split_sizes, in_tensor)
        if out_tensor.numel() == 0:
            out_tensor = torch.empty([1], device=out_tensor.device, dtype=out_tensor.dtype)
        output_lengths, output_offsets = computeLengthsAndOffsets(output_split_sizes, out_tensor)
        # out_tensor.record_stream(stream)  # avoid allocator mem reuse
        self.nccl.ncclAllToAllv(
            buffer_type(in_tensor.data_ptr()),
            input_lengths,
            input_offsets,
            buffer_type(out_tensor.data_ptr()),
            output_lengths,
            output_offsets,
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        # stream.synchronize() # sync the stream to make sure the all_to_all is finished


class NCCLLibrary:
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t  ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t  ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        # note that ncclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "ncclCommInitRank", ncclResult_t, [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int]
        ),
        # ncclResult_t  ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        # note that cudaStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t],
        ),
        # ncclResult_t  ncclAllToAllv(
        #     buffer_type sendbuff, const int *sendcounts, const int *sendsizes,
        #     buffer_type recvbuff, const int *recvcounts, const int *recvsizes,
        #     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream,);
        # Function("ncclAllToAllv", ncclResult_t, [
        #     buffer_type, ctypes.POINTER(ctypes.c_int),
        #     ctypes.POINTER(ctypes.c_int), buffer_type,
        #     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        #     ncclDataType_t, ncclComm_t, cudaStream_t
        # ]),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ncclResult_t  ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    ]

    exported_custom_functions = [
        Function(
            "CustomNcclAllToAllv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                buffer_type,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        Function(
            "CustomNcclAlltoAll",
            ncclResult_t,
            [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None, custom_nccl_so_file: Optional[str] = None):
        so_file = so_file
        custom_nccl_so_file = custom_nccl_so_file

        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            print(
                "Failed to load NCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        try:
            if custom_nccl_so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(custom_nccl_so_file)
                NCCLLibrary.path_to_library_cache[custom_nccl_so_file] = lib
            self.custom_alltoall_lib = NCCLLibrary.path_to_library_cache[custom_nccl_so_file]
        except Exception as e:
            print(
                "Failed to load NCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable VLLM_NCCL_SO_PATH"
                " to point to the correct nccl library path.",
                custom_nccl_so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs

        if custom_nccl_so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in NCCLLibrary.exported_custom_functions:
                f = getattr(self.custom_alltoall_lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[custom_nccl_so_file] = _funcs

        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file] | NCCLLibrary.path_to_dict_mapping[custom_nccl_so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank))
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # `datatype` actually should be `ncclDataType_t`
        # and `op` should be `ncclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.NCCL_CHECK(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def ncclAllToAll(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ):
        self.NCCL_CHECK(
            self._funcs["CustomNcclAlltoAll"](
                sendbuff, recvbuff, count, datatype, ncclRedOpTypeEnum.ncclSum, comm, stream
            )
        )

    def ncclAllToAllv(
        self,
        sendbuff: buffer_type,
        sendcounts: List[int],
        sendsizes: List[int],
        recvbuff: buffer_type,
        recvcounts: List[int],
        recvsizes: List[int],
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        # convert list to ctypes array
        sendcounts = (ctypes.c_int * len(sendcounts))(*sendcounts)
        sendsizes = (ctypes.c_int * len(sendsizes))(*sendsizes)
        recvcounts = (ctypes.c_int * len(recvcounts))(*recvcounts)
        recvsizes = (ctypes.c_int * len(recvsizes))(*recvsizes)
        self.NCCL_CHECK(
            self._funcs["CustomNcclAllToAllv"](
                sendbuff, sendcounts, sendsizes, recvbuff, recvcounts, recvsizes, datatype, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))
