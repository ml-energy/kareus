#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <mscclpp/core.hpp>

#include "comm_wrapper.cuh"

extern void init_bootstrap(int rank, int nranks);
extern mscclpp::UniqueId get_unique_id();
extern std::shared_ptr<CommunicatorWrapper> mscclppCommInitRank(int world_size, mscclpp::UniqueId uniqueId, int rank);
namespace py = pybind11;

extern void Custom_MScclpp_AlltoAll(std::shared_ptr<CommunicatorWrapper> comm,
                                    void* input_buff,
                                    void* output_buff,
                                    const size_t buff_size,
                                    cudaStream_t stream,
                                    int sm_num,
                                    int block_size,
                                    int nranks,
                                    int rank);

extern void Custom_MScclpp_UnevenAlltoAll(std::shared_ptr<CommunicatorWrapper> comm,
                                          void* input_buff,
                                          void* output_buff,
                                          const size_t buff_size,
                                          cudaStream_t stream,
                                          int sm_num,
                                          int block_size,
                                          const int nranks,
                                          const int rank,
                                          std::vector<int>& ranks_send,
                                          std::vector<int>& ranks_recv);

extern void Custom_MScclpp_AlltoAllv(
    std::shared_ptr<CommunicatorWrapper> comm,
    void* input_buff,
    void* output_buff,
    const size_t input_size,   // input_buff_size = input_size * sizeof(cutlass::half_t)
    const size_t output_size,  // output_buff_size = output_size * sizeof(cutlass::half_t)
    cudaStream_t stream,
    int sm_num,
    int block_size,
    const int nranks,
    const int rank,
    std::vector<int>& input_lengths,
    std::vector<int>& input_offsets,
    std::vector<std::vector<int>>& output_lengths_all,
    std::vector<std::vector<int>>& output_offsets_all);

extern void cached_Custom_MScclpp_AlltoAllv(std::shared_ptr<CommunicatorWrapper> comm,
                                            void* input_buff,
                                            void* output_buff,
                                            const size_t input_size,
                                            const size_t output_size,
                                            cudaStream_t stream,
                                            int sm_num,
                                            int block_size,
                                            std::vector<int>& input_lengths,
                                            std::vector<int>& input_offsets,
                                            std::vector<std::vector<int>>& output_lengths_all,
                                            std::vector<std::vector<int>>& output_offsets_all);

extern void cached_Custom_MScclpp_AlltoAllv_FuseCopy(std::shared_ptr<CommunicatorWrapper> comm,
                                                     void* input_buff,
                                                     void* output_buff,
                                                     const size_t input_size,
                                                     const size_t output_size,
                                                     cudaStream_t stream,
                                                     int sm_num,
                                                     int block_size,
                                                     std::vector<int>& input_lengths,
                                                     std::vector<int>& input_offsets,
                                                     std::vector<std::vector<int>>& output_lengths_all,
                                                     std::vector<std::vector<int>>& output_offsets_all);

extern void Custom_MScclpp_AllReduce(std::shared_ptr<CommunicatorWrapper> comm,
                                     cudaStream_t stream,
                                     int sm_num,
                                     int block_size);

extern void Custom_MScclpp_AllReduce_FuseCopy(std::shared_ptr<CommunicatorWrapper> comm,
                                              void* input_buff,
                                              void* output_buff,
                                              const size_t input_size,
                                              const size_t output_size,
                                              cudaStream_t stream,
                                              int sm_num,
                                              int block_size);

extern void Custom_MScclpp_AllReduce_BF16(std::shared_ptr<CommunicatorWrapper> comm,
                                          cudaStream_t stream,
                                          int sm_num,
                                          int block_size);

extern void Custom_MScclpp_AllReduce_FuseCopy_BF16(std::shared_ptr<CommunicatorWrapper> comm,
                                                    void* input_buff,
                                                    void* output_buff,
                                                    const size_t input_size,
                                                    const size_t output_size,
                                                    cudaStream_t stream,
                                                    int sm_num,
                                                    int block_size);

extern void init_NetAlltoAllv_wrapper(std::shared_ptr<CommunicatorWrapper> comm,
                                      const int rank,
                                      const int nranks,
                                      cudaStream_t stream);

extern void init_NetAllReduce_wrapper_cached(std::shared_ptr<CommunicatorWrapper> comm,
                                            // void* input_buff,
                                            // void* output_buff,
                                            // const size_t tensor_size,
                                            const int rank,
                                            const int nranks,
                                            cudaStream_t stream);

extern void init_NetAllReduce_wrapper(std::shared_ptr<CommunicatorWrapper> comm,
                                      void* input_buff,
                                      void* output_buff,
                                      const size_t tensor_size,
                                      const int rank,
                                      const int nranks,
                                      cudaStream_t stream);

extern void init_NetAllReduce_wrapper_bf16(std::shared_ptr<CommunicatorWrapper> comm,
                                          void* input_buff,
                                          void* output_buff,
                                          const size_t tensor_size,
                                          const int rank,
                                          const int nranks,
                                          cudaStream_t stream);

extern void init_NetAllReduce_wrapper_cached_bf16(std::shared_ptr<CommunicatorWrapper> comm,
                                                   // void* input_buff,
                                                   // void* output_buff,
                                                   // const size_t tensor_size,
                                                   const int rank,
                                                   const int nranks,
                                                   cudaStream_t stream);

PYBIND11_MODULE(msccl_comm, m) {
  m.doc() = "Python bindings for MSCCL communication primitives";

  // Bind the initialization functions
  m.def("init_bootstrap", &init_bootstrap, "Initialize MSCCL bootstrap", py::arg("rank"), py::arg("nranks"));

  //   m.def("get_unique_id", &get_unique_id, "Get a unique ID for
  //   communication");
  // In Python binding
  m.def("get_unique_id", []() {
    auto id = get_unique_id();
    return py::bytes((char*)id.data(), MSCCLPP_UNIQUE_ID_BYTES);
  });

  // Bind the Communicator initialization
  // py::class_<mscclpp::Communicator, std::shared_ptr<mscclpp::Communicator>>(m, "Communicator");
  py::class_<CommunicatorWrapper, std::shared_ptr<CommunicatorWrapper>>(m, "Communicator");

  m.def(
      "init_communicator",
      [](int world_size, py::bytes unique_id_bytes, int rank) {
        char* buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(unique_id_bytes.ptr(), &buffer, &length)) {
          throw std::runtime_error("Failed to get bytes data");
        }
        if (length != MSCCLPP_UNIQUE_ID_BYTES) {
          throw std::runtime_error("Invalid unique_id length");
        }

        mscclpp::UniqueId unique_id;
        std::memcpy(unique_id.data(), buffer, MSCCLPP_UNIQUE_ID_BYTES);

        return mscclppCommInitRank(world_size, unique_id, rank);
      },
      "Initialize MSCCL communicator");

  // Bind the AlltoAll function
  m.def(
      "msccl_alltoall",
      [](std::shared_ptr<CommunicatorWrapper> comm,
         uintptr_t input_buff,
         uintptr_t output_buff,
         size_t buff_size,
         uintptr_t stream_ptr,
         int sm_num,
         int block_size,
         int nranks,
         int rank) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
        void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
        Custom_MScclpp_AlltoAll(
            comm, input_buff_ptr, output_buff_ptr, buff_size, stream, sm_num, block_size, nranks, rank);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
      },
      "Perform custom AlltoAll communication");

  m.def(
      "msccl_alltoall_uneven",
      [](std::shared_ptr<CommunicatorWrapper> comm,
         uintptr_t input_buff,
         uintptr_t output_buff,
         size_t buff_size,
         uintptr_t stream_ptr,
         int sm_num,
         int block_size,
         int nranks,
         int rank,
         std::vector<int> ranks_send,
         std::vector<int> ranks_recv) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
        void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
        Custom_MScclpp_UnevenAlltoAll(comm,
                                      input_buff_ptr,
                                      output_buff_ptr,
                                      buff_size,
                                      stream,
                                      sm_num,
                                      block_size,
                                      nranks,
                                      rank,
                                      ranks_send,
                                      ranks_recv);
      },
      "Perform custom AlltoAll communication");

  // deprecated
  m.def(
      "msccl_alltoallv",
      [](std::shared_ptr<CommunicatorWrapper> comm,
         uintptr_t input_buff,
         uintptr_t output_buff,
         size_t input_size,
         size_t output_size,
         uintptr_t stream_ptr,
         int sm_num,
         int block_size,
         int nranks,
         int rank,
         std::vector<int> input_lengths,
         std::vector<int> input_offsets,
         std::vector<std::vector<int>> output_lengths_all,
         std::vector<std::vector<int>> output_offsets_all) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
        void* output_buff_ptr = reinterpret_cast<void*>(output_buff);

        Custom_MScclpp_AlltoAllv(comm,
                                 input_buff_ptr,
                                 output_buff_ptr,
                                 input_size,
                                 output_size,
                                 stream,
                                 sm_num,
                                 block_size,
                                 nranks,
                                 rank,
                                 input_lengths,
                                 input_offsets,
                                 output_lengths_all,
                                 output_offsets_all);
      },
      "Perform custom AlltoAllv communication");

  m.def(
      "msccl_alltoallv_cached",
      [](std::shared_ptr<CommunicatorWrapper> comm,
         uintptr_t input_buff,
         uintptr_t output_buff,
         size_t input_size,
         size_t output_size,
         uintptr_t stream_ptr,
         int sm_num,
         int block_size,
         //  int nranks,
         //  int rank,
         std::vector<int> input_lengths,
         std::vector<int> input_offsets,
         std::vector<std::vector<int>> output_lengths_all,
         std::vector<std::vector<int>> output_offsets_all) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
        void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
        cached_Custom_MScclpp_AlltoAllv_FuseCopy(comm,
                                                 input_buff_ptr,
                                                 output_buff_ptr,
                                                 input_size,
                                                 output_size,
                                                 stream,
                                                 sm_num,
                                                 block_size,
                                                 // nranks,
                                                 // rank,
                                                 input_lengths,
                                                 input_offsets,
                                                 output_lengths_all,
                                                 output_offsets_all);
      },
      "Perform custom AlltoAllv communication");
  
  m.def(
    "msccl_allreduce",
    [](std::shared_ptr<CommunicatorWrapper> comm,
        uintptr_t stream_ptr,
        int sm_num,
        int block_size) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      Custom_MScclpp_AllReduce(comm, stream, sm_num, block_size);
    },
    "Perform custom AllReduce communication");

  m.def(
    "msccl_allreduce_bf16",
    [](std::shared_ptr<CommunicatorWrapper> comm,
        uintptr_t stream_ptr,
        int sm_num,
        int block_size) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      Custom_MScclpp_AllReduce_BF16(comm, stream, sm_num, block_size);
    },
    "Perform custom AllReduce communication for BF16");

  m.def(
    "msccl_allreduce_cached",
    [](std::shared_ptr<CommunicatorWrapper> comm,
        uintptr_t input_buff,
        uintptr_t output_buff,
        size_t input_size,
        size_t output_size,
        uintptr_t stream_ptr,
        int sm_num,
        int block_size) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
      void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
      Custom_MScclpp_AllReduce_FuseCopy(
        comm, input_buff_ptr, output_buff_ptr, input_size, output_size, stream, sm_num, block_size);
    },
    "Perform custom AllReduce communication");

  m.def(
    "msccl_allreduce_cached_bf16",
    [](std::shared_ptr<CommunicatorWrapper> comm,
        uintptr_t input_buff,
        uintptr_t output_buff,
        size_t input_size,
        size_t output_size,
        uintptr_t stream_ptr,
        int sm_num,
        int block_size) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
      void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
      Custom_MScclpp_AllReduce_FuseCopy_BF16(
        comm, input_buff_ptr, output_buff_ptr, input_size, output_size, stream, sm_num, block_size);
    },
    "Perform custom AllReduce communication for BF16");

  m.def(
      "init_NetAlltoAllv_wrapper",
      [](std::shared_ptr<CommunicatorWrapper> comm, int rank, int nranks, uintptr_t stream_ptr) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        init_NetAlltoAllv_wrapper(comm, rank, nranks, stream);
      },
      "Initialize NetAlltoAllv wrapper");
  
  m.def(
    "init_NetAllReduce_wrapper",
    [](std::shared_ptr<CommunicatorWrapper> comm, uintptr_t input_buff, uintptr_t output_buff, size_t tensor_size, int rank, int nranks, uintptr_t stream_ptr) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
      void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
      init_NetAllReduce_wrapper(comm, input_buff_ptr, output_buff_ptr, tensor_size, rank, nranks, stream);
    },
    "Initialize NetAllReduce wrapper with input/output buffers");
  m.def(
    "init_NetAllReduce_wrapper_cached",
    [](std::shared_ptr<CommunicatorWrapper> comm, int rank, int nranks, uintptr_t stream_ptr) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      
      init_NetAllReduce_wrapper_cached(comm, rank, nranks, stream);
    },
    "Initialize NetAllReduce wrapper with internal buffers");

  m.def(
    "init_NetAllReduce_wrapper_bf16",
    [](std::shared_ptr<CommunicatorWrapper> comm, uintptr_t input_buff, uintptr_t output_buff, size_t tensor_size, int rank, int nranks, uintptr_t stream_ptr) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      void* input_buff_ptr = reinterpret_cast<void*>(input_buff);
      void* output_buff_ptr = reinterpret_cast<void*>(output_buff);
      init_NetAllReduce_wrapper_bf16(comm, input_buff_ptr, output_buff_ptr, tensor_size, rank, nranks, stream);
    },
    "Initialize NetAllReduce BF16 wrapper with input/output buffers");
  m.def(
    "init_NetAllReduce_wrapper_cached_bf16",
    [](std::shared_ptr<CommunicatorWrapper> comm, int rank, int nranks, uintptr_t stream_ptr) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      init_NetAllReduce_wrapper_cached_bf16(comm, rank, nranks, stream);
    },
    "Initialize NetAllReduce BF16 wrapper with internal buffers");
}
