#pragma once
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

#include "comm.h"
#include "config.h"
#include "sleep.cuh"
#include "tensor.cuh"
// #include "vortexData.cuh"
#include <assert.h>

#ifdef ENABLE_MPI
// use mpi to bcast the handles
#include <mpi.h>
#else
// use thread sync to get the handles
#include "networkManager.cuh"
#endif

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "operatorWrapper.cuh"
#include <cutlass/bfloat16.h>

class NetWrapper : public OperatorWrapper {
  public:
  using Element = cutlass::half_t;
  Element* input = nullptr;
  Element* output = nullptr;
  pllmTensor<Element> pllm_tensor_input;
  pllmTensor<Element> pllm_tensor_output;
  size_t input_size;
  size_t output_size;
  size_t rank, nranks;
  bool isInitialized = false;

  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelHandles;

  NetWrapper() {}
  void checkassert() {
    assert(input_size % nranks == 0);
    assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
    assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);
  }

  pllmTensor<Element> getInput() {
    return pllm_tensor_input;
  }
  pllmTensor<Element> getOutput() {
    return pllm_tensor_output;
  }
  int nblocks, nthreads;
  bool sync_mode;
  virtual NetWrapper& configRun(int nblocks, int nthreads, bool sync) {
    this->nblocks = nblocks;
    this->nthreads = nthreads;
    this->sync_mode = sync;
    return *this;
  }

  OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
    log_tensor(logger, name + "input", getInput(), 10, 20);
    log_tensor(logger, name + "output", getOutput(), 10, 20);
    return *this;
  }

  protected:
  void init(int rank, int nranks, Element* input, Element* output, size_t input_size, size_t output_size) {
    this->isInitialized = true;
    this->rank = rank;
    this->nranks = nranks;
    this->input = input;
    this->input_size = input_size;
    this->output = output;
    this->output_size = output_size;
    checkassert();
  }
  virtual void work() override {
    operator()(stream, nblocks, nthreads, sync_mode);
  }
  virtual void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {}

  virtual void sync(cudaStream_t stream) {}

  void setupSmChannels(std::shared_ptr<mscclpp::Communicator> comm,
                       std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                       mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
                       Element* input,
                       Element* output,
                       size_t input_size,
                       size_t output_size) {
    // Registers input buffer memory using CUDA IPC transport for inter-GPU communication.
    const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
    mscclpp::RegisteredMemory inputBuffRegMem =
        comm->registerMemory(input, input_size * sizeof(Element), allTransports);
    mscclpp::RegisteredMemory outputBuffRegMem;
    // If input and output are different, it also registers the output buffer.
    if (input != output)
      outputBuffRegMem = comm->registerMemory(output, output_size * sizeof(Element), allTransports);

    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
    mscclpp::RegisteredMemory& localRegMemory = (input != output) ? outputBuffRegMem : inputBuffRegMem;

    // Exchange registered memory information with all other GPUs (nranks - 1).
    for (size_t r = 0; r < nranks; ++r) {
      if (r == rank)
        continue;
      comm->sendMemoryOnSetup(localRegMemory, r, 0);      // Send its local registered memory to peer ranks
      auto remoteMemory = comm->recvMemoryOnSetup(r, 0);  // Receive remote registered memory from peer ranks
      remoteRegMemories.push_back(remoteMemory);
    }
    comm->setup();

    std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
    for (size_t i = 0; i < connections.size(); ++i) {
      // Create synchronization mechanisms (Semaphore) between GPUs
      smSemaphores.emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[i]));
    }
    comm->setup();
    for (size_t i = 0; i < connections.size(); ++i) {
      // Create SmChannel between GPUs (communication link between the local GPU and a remote peer)
      // smChannels store a semaphore for synchronization, a remote memory buffer for reading from peer GPUs, and a
      // local buffer for writing data
      smChannels.emplace_back(smSemaphores[i], remoteRegMemories[i].get(), inputBuffRegMem.data());
      smChannelHandles.emplace_back(mscclpp::deviceHandle(smChannels.back()));
    }
    comm->setup();

    // Allocates device memory for smChannelHandlesCuda and copies smChannelHandles to device memory
    assert(connections.size() == nranks - 1);
    CUDA_CHECK(cudaMallocAsync(
        smChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>), stream));
    CUDA_CHECK(cudaMemcpyAsync(*smChannelHandlesCuda,
                               &smChannelHandles[smChannelHandles.size() - (nranks - 1)],
                               (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>),
                               cudaMemcpyHostToDevice,
                               stream));
  }

  // Convenience overload: same buffer for input/output
  void setupSmChannels(std::shared_ptr<mscclpp::Communicator> comm,
                       std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                       mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
                       Element* buff,
                       size_t buff_size) {
    setupSmChannels(comm, connections, smChannelHandlesCuda, buff, buff, buff_size, buff_size);
  }

  void setupSmChannelsBytes(std::shared_ptr<mscclpp::Communicator> comm,
                            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                            mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
                            void* input,
                            void* output,
                            size_t input_elems,
                            size_t output_elems,
                            size_t element_size_bytes) {
    const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
    mscclpp::RegisteredMemory inputBuffRegMem =
        comm->registerMemory(input, input_elems * element_size_bytes, allTransports);
    mscclpp::RegisteredMemory outputBuffRegMem;
    if (input != output)
      outputBuffRegMem = comm->registerMemory(output, output_elems * element_size_bytes, allTransports);

    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
    mscclpp::RegisteredMemory& localRegMemory = (input != output) ? outputBuffRegMem : inputBuffRegMem;

    for (size_t r = 0; r < nranks; ++r) {
      if (r == rank) continue;
      comm->sendMemoryOnSetup(localRegMemory, r, 0);
      auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
      remoteRegMemories.push_back(remoteMemory);
    }
    comm->setup();

    std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
    for (size_t i = 0; i < connections.size(); ++i) {
      smSemaphores.emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[i]));
    }
    comm->setup();
    for (size_t i = 0; i < connections.size(); ++i) {
      smChannels.emplace_back(smSemaphores[i], remoteRegMemories[i].get(), inputBuffRegMem.data());
      smChannelHandles.emplace_back(mscclpp::deviceHandle(smChannels.back()));
    }
    comm->setup();

    assert(connections.size() == nranks - 1);
    CUDA_CHECK(cudaMallocAsync(
        smChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>), stream));
    CUDA_CHECK(cudaMemcpyAsync(*smChannelHandlesCuda,
                               &smChannelHandles[smChannelHandles.size() - (nranks - 1)],
                               (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>),
                               cudaMemcpyHostToDevice,
                               stream));
  }
};

class NetAllGather : public NetWrapper {
  public:
  bool columnwise = false;
  mscclpp::DeviceSyncer* syncersCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannelHandlesCuda;

  NetAllGather() : NetWrapper() {}
  // Init AllGather using explicit input and output buffers.
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,
            pllmTensor<Element> output) {
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMalloc(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer)));
    CUDA_CHECK(cudaMemcpy(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));

    setupSmChannels(comm, connections, &smChannelHandlesCuda, input.ptr, output.ptr, input.size(), output.size());
  }

  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> buff) {
    init(comm, connections, rank, nranks, buff, buff);
  }

  ~NetAllGather() {
    CUDA_CHECK(cudaFree(syncersCuda));
    CUDA_CHECK(cudaFree(smChannelHandlesCuda));
  }

  NetAllGather& setColumnwise(bool columnwise = true) {
    this->columnwise = columnwise;
    return *this;
  }

  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {
    const int nchannels = nranks - 1;
    if (!columnwise) {
      assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2);
      const uint64_t nelem_per_shard = input_size / nranks;
      const uint64_t local_offset = rank * nelem_per_shard;
      if (sync) {
        allgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
            smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
      } else {
        allgatherKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
            smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
      }
    } else {
      assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2 / nranks);
      const uint64_t input_ncols = this->pllm_tensor_input.dim2;
      const uint64_t output_ncols = this->pllm_tensor_output.dim2;
      const uint64_t output_row_offset = rank * input_ncols;
      const uint64_t nrows = this->pllm_tensor_input.dim1;

      constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
      assert(input_ncols % n_half_per_int4 == 0);
      assert(output_ncols % n_half_per_int4 == 0);
      assert(output_row_offset % n_half_per_int4 == 0);
      assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
      assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);

      columnwiseAllgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(smChannelHandlesCuda,
                                                                            syncersCuda,
                                                                            sync,
                                                                            nchannels,
                                                                            input_ncols,
                                                                            output_ncols,
                                                                            output_row_offset,
                                                                            nrows,
                                                                            input,
                                                                            output);
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smChannelHandlesCuda, nchannels);
  }

  // override copy constructor
  NetAllGather(const NetAllGather& other) = delete;
};

class NetAlltoAll : public NetWrapper {
  public:
  bool columnwise = false;
  mscclpp::DeviceSyncer* syncersCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannelHandlesCuda;

  NetAlltoAll() : NetWrapper() {}
  // Init Alltoall using explicit input and output buffers.
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,     // [buffer_size, 1]
            pllmTensor<Element> output) {  // [buffer_size, 1]
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMallocAsync(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice, stream));

    assert(input.ptr != output.ptr);
    setupSmChannels(comm, connections, &smChannelHandlesCuda, input.ptr, output.ptr, input.size(), output.size());
  }

  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> buff) {
    init(comm, connections, rank, nranks, buff, buff);
  }

  ~NetAlltoAll() {
    CUDA_CHECK(cudaFreeAsync(syncersCuda, stream));
    CUDA_CHECK(cudaFreeAsync(smChannelHandlesCuda, stream));
  }

  NetAlltoAll& setColumnwise(bool columnwise = true) {
    this->columnwise = columnwise;
    return *this;
  }

  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {
    assert(columnwise == False);
    const int nchannels = nranks - 1;
    if (!columnwise) {
      assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2);
      const uint64_t nelem_per_shard = input_size / nranks;
      const uint64_t local_offset = rank * nelem_per_shard;
      if (sync) {
        alltoallKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
            smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
      } else {
        alltoallKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
            smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard, input, output);
      }
    } else {
      // assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      // assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2 / nranks);
      // const uint64_t input_ncols = this->pllm_tensor_input.dim2;
      // const uint64_t output_ncols = this->pllm_tensor_output.dim2;
      // const uint64_t output_row_offset = rank * input_ncols;
      // const uint64_t nrows = this->pllm_tensor_input.dim1;

      // constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
      // assert(input_ncols % n_half_per_int4 == 0);
      // assert(output_ncols % n_half_per_int4 == 0);
      // assert(output_row_offset % n_half_per_int4 == 0);
      // assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
      // assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);

      // columnwiseAllgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
      //     smChannelHandlesCuda, syncersCuda, sync, nchannels,
      //     input_ncols, output_ncols, output_row_offset, nrows, input, output);
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smChannelHandlesCuda, nchannels);
  }

  // override copy constructor
  NetAlltoAll(const NetAllGather& other) = delete;
};

class NetAlltoAllUneven : public NetWrapper {
  public:
  bool columnwise = false;
  mscclpp::DeviceSyncer* syncersCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannelHandlesCuda;
  int64_t* local_offsets;
  int64_t* remote_offsets;
  size_t nelem_per_shard;
  bool if_recv;

  NetAlltoAllUneven() : NetWrapper() {}

  // Init Alltoall using explicit input and output buffers.
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,   // [input_buffer_size, 1]
            pllmTensor<Element> output,  // [output_buffer_size, 1]
            std::vector<int>& ranks_send,
            std::vector<int>& ranks_recv) {
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());
    initUneven(ranks_send, ranks_recv);

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMallocAsync(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice, stream));

    assert(input.ptr != output.ptr);
    setupSmChannels(comm, connections, &smChannelHandlesCuda, input.ptr, output.ptr, input.size(), output.size());
  }

  // void init(std::shared_ptr<mscclpp::Communicator> comm,
  //                 std::vector<std::shared_ptr<mscclpp::Connection>> connections,
  //                 int rank,
  //                 int nranks,
  //                 pllmTensor<Element> buff) {
  //         init(comm, connections, rank, nranks, buff, buff);
  // }

  void initUneven(std::vector<int>& ranks_send, std::vector<int>& ranks_recv) {
    // TODO: assert divisible
    this->nelem_per_shard = input_size * ranks_recv.size() / ranks_send.size();
    assert(nelem_per_shard == (output_size / ranks_recv.size()));

    std::vector<int64_t> local_offset_host(nranks);
    for (int i = 0; i < nranks; i++) {
      int rank_idx_in_send = rank_idx(i, ranks_send);
      local_offset_host[i] = (int64_t)rank_idx_in_send * (int64_t)nelem_per_shard;
    }
    CUDA_CHECK(cudaMallocAsync(&this->local_offsets, nranks * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        this->local_offsets, local_offset_host.data(), nranks * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    std::vector<int64_t> remote_offset_host(nranks);
    for (int i = 0; i < nranks; i++) {
      int rank_idx_in_recv = rank_idx(i, ranks_recv);
      remote_offset_host[i] = (int64_t)rank_idx_in_recv * (int64_t)nelem_per_shard;
      if (i == rank) {
        this->if_recv = rank_idx_in_recv > 0;
      }
    }
    CUDA_CHECK(cudaMallocAsync(&this->remote_offsets, nranks * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        this->remote_offsets, remote_offset_host.data(), nranks * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
  }

  int rank_idx(int rank, std::vector<int>& ranks) {
    for (int i = 0; i < ranks.size(); i++) {
      if (rank == ranks[i]) {
        return i;
      }
    }
    return -1;
  }

  // // Override setupSmChannels
  // void setupSmChannels(std::shared_ptr<mscclpp::Communicator> comm,
  //                      std::vector<std::shared_ptr<mscclpp::Connection>> connections,
  //                      mscclpp::DeviceHandle<mscclpp::SmChannel>** smChannelHandlesCuda,
  //                      Element* input, Element* output, size_t input_size, size_t output_size) {
  //     // Registers input buffer memory using CUDA IPC transport for inter-GPU communication.
  //     const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;

  //     mscclpp::RegisteredMemory inputBuffRegMem;
  //     if (this->if_send) {
  //         inputBuffRegMem = comm->registerMemory(input, input_size * sizeof(Element), allTransports);
  //     }
  //     mscclpp::RegisteredMemory outputBuffRegMem;
  //     if (this->if_recv) {
  //         outputBuffRegMem = comm->registerMemory(output, output_size * sizeof(Element), allTransports);
  //     }

  //     std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories(nranks - 1);
  //     mscclpp::RegisteredMemory& localRegMemory;
  //     if (this->if_recv) {
  //         localRegMemory = outputBuffRegMem;
  //     }

  //     // Exchange registered memory information with all other GPUs (nranks - 1).
  //     int peer = 0;
  //     for (size_t r = 0; r < nranks; ++r) {
  //         if (r == rank) continue;
  //         bool peer_recv = this->rank_idx_in_recv[r] > 0;
  //         if (peer_recv && this->if_send) {
  //             comm->sendMemoryOnSetup(localRegMemory, r, 0); // Send its local registered memory to peer ranks
  //         }
  //         bool peer_send = this->rank_idx_in_send[r] > 0;
  //         if (this->if_recv && peer_send) {
  //             auto remoteMemory = comm->recvMemoryOnSetup(r, 0); // Receive remote registered memory from peer ranks
  //             // remoteRegMemories.push_back(remoteMemory);
  //             remoteRegMemories[peer] = remoteMemory;
  //         }
  //         peer++;
  //     }
  //     comm->setup();

  //     std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  //     for (size_t i = 0; i < connections.size(); ++i) {
  //         // Create synchronization mechanisms (Semaphore) between GPUs
  //         smSemaphores.emplace_back(std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*comm, connections[i]));
  //     }
  //     comm->setup();
  //     for (size_t i = 0; i < connections.size(); ++i) {
  //         // Create SmChannel between GPUs (communication link between the local GPU and a remote peer)
  //         // smChannels store a semaphore for synchronization, a remote memory buffer for reading from peer GPUs, and
  //         a local buffer for writing data smChannels.emplace_back(smSemaphores[i], remoteRegMemories[i].get(),
  //         inputBuffRegMem.data()); smChannelHandles.emplace_back(mscclpp::deviceHandle(smChannels.back()));
  //     }
  //     comm->setup();

  //     // Allocates device memory for smChannelHandlesCuda and copies smChannelHandles to device memory
  //     assert(connections.size() == nranks - 1);
  //     CUDA_CHECK(cudaMalloc(smChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>)));
  //     CUDA_CHECK(cudaMemcpy(*smChannelHandlesCuda, &smChannelHandles[smChannelHandles.size() - (nranks - 1)],
  //                           (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SmChannel>),
  //                           cudaMemcpyHostToDevice));
  // }

  ~NetAlltoAllUneven() {
    CUDA_CHECK(cudaFreeAsync(syncersCuda, stream));
    CUDA_CHECK(cudaFreeAsync(smChannelHandlesCuda, stream));
    CUDA_CHECK(cudaFreeAsync(local_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(remote_offsets, stream));
  }

  NetAlltoAllUneven& setColumnwise(bool columnwise = true) {
    this->columnwise = columnwise;
    return *this;
  }

  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {
    assert(sync == True);
    assert(columnwise == False);
    const int nchannels = nranks - 1;
    if (!columnwise) {
      // assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      // assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2);
      // const uint64_t nelem_per_shard = input_size / nranks;
      // const int64_t local_offset = this->rank_idx_in_send[rank] * nelem_per_shard;
      if (sync) {
        alltoallUnenvenKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(smChannelHandlesCuda,
                                                                          syncersCuda,
                                                                          rank,
                                                                          nchannels,
                                                                          local_offsets,
                                                                          remote_offsets,
                                                                          nelem_per_shard,
                                                                          input,
                                                                          output);
      } else {
        // alltoallKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
        //     smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard_, input, output);
      }
    } else {
      // assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      // assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2 / nranks);
      // const uint64_t input_ncols = this->pllm_tensor_input.dim2;
      // const uint64_t output_ncols = this->pllm_tensor_output.dim2;
      // const uint64_t output_row_offset = rank * input_ncols;
      // const uint64_t nrows = this->pllm_tensor_input.dim1;

      // constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
      // assert(input_ncols % n_half_per_int4 == 0);
      // assert(output_ncols % n_half_per_int4 == 0);
      // assert(output_row_offset % n_half_per_int4 == 0);
      // assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
      // assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);

      // columnwiseAllgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
      //     smChannelHandlesCuda, syncersCuda, sync, nchannels,
      //     input_ncols, output_ncols, output_row_offset, nrows, input, output);
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smChannelHandlesCuda, nchannels);
  }

  // override copy constructor
  NetAlltoAllUneven(const NetAllGather& other) = delete;
};

class NetAlltoAllv : public NetWrapper {
  public:
  bool columnwise = false;
  mscclpp::DeviceSyncer* syncersCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannelHandlesCuda;
  int* input_lengths = nullptr;
  int* input_offsets = nullptr;
  int* output_lengths_all = nullptr;
  int* output_offsets_all = nullptr;
  Element* tmp_input_tensor_ptr = nullptr;
  Element* tmp_output_tensor_ptr = nullptr;
  bool fuse_copy_mode = false;

  NetAlltoAllv() : NetWrapper() {}

  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,  // [input_buffer_size, 1]
            pllmTensor<Element> output  // [output_buffer_size, 1]
  ) {
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMallocAsync(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice, stream));

    assert(input.ptr != output.ptr);
    setupSmChannels(comm, connections, &smChannelHandlesCuda, input.ptr, output.ptr, input.size(), output.size());
  }

  // Init Alltoall using explicit input and output buffers.
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,   // [input_buffer_size, 1]
            pllmTensor<Element> output,  // [output_buffer_size, 1]
            std::vector<int>& input_lengths,
            std::vector<int>& input_offsets,
            std::vector<std::vector<int>>& output_lengths_all,
            std::vector<std::vector<int>>& output_offsets_all) {
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());
    initAlltoallv(input_lengths, input_offsets, output_lengths_all, output_offsets_all);

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMallocAsync(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice, stream));

    assert(input.ptr != output.ptr);
    setupSmChannels(comm, connections, &smChannelHandlesCuda, input.ptr, output.ptr, input.size(), output.size());
  }

  void initAlltoallv(std::vector<int>& input_lengths,
                     std::vector<int>& input_offsets,
                     std::vector<std::vector<int>>& output_lengths_all,
                     std::vector<std::vector<int>>& output_offsets_all) {
    assert(input_lengths.size() == nranks);
    assert(input_offsets.size() == nranks);
    assert(output_lengths_all.size() == nranks);
    assert(output_offsets_all.size() == nranks);
    for (int i = 0; i < nranks; i++) {
      assert(output_lengths_all[i].size() == nranks);
      assert(output_offsets_all[i].size() == nranks);
    }

    int data_size = nranks * 2 + nranks * (2 * nranks);
    std::vector<int> data_host(data_size);
    // input_lengths
    for (int i = 0; i < nranks; i++) {
      data_host[i] = input_lengths[i];
    }
    // input_offsets
    for (int i = 0; i < nranks; i++) {
      data_host[nranks + i] = input_offsets[i];
    }
    // output_lengths
    for (int i = 0; i < nranks; i++) {
      for (int j = 0; j < nranks; j++) {
        data_host[2 * nranks + i * nranks + j] = output_lengths_all[i][j];
      }
    }
    // output_offsets
    for (int i = 0; i < nranks; i++) {
      for (int j = 0; j < nranks; j++) {
        data_host[2 * nranks + nranks * nranks + i * nranks + j] = output_offsets_all[i][j];
      }
    }

    if (this->input_lengths == nullptr) {
      int* data_ptr;
      CUDA_CHECK(cudaMallocAsync(&data_ptr, data_size * sizeof(int), this->stream));  // stream is set by setStream
      CUDA_CHECK(
          cudaMemcpyAsync(data_ptr, data_host.data(), data_size * sizeof(int), cudaMemcpyHostToDevice, this->stream));

      this->input_lengths = data_ptr;
      this->input_offsets = data_ptr + nranks;
      this->output_lengths_all = data_ptr + 2 * nranks;
      this->output_offsets_all = data_ptr + 2 * nranks + nranks * nranks;
    } else {
      CUDA_CHECK(cudaMemcpyAsync(
          this->input_lengths, data_host.data(), data_size * sizeof(int), cudaMemcpyHostToDevice, this->stream));
    }
  }

  // here we add some new fixed parameters for netwrapper to reuse them without high cost init
  void update_wrapper(int rank,
                      int nranks,
                      void* input_tensor_ptr,
                      void* output_tensor_ptr,
                      size_t input_size,
                      size_t output_size,
                      cudaStream_t stream,
                      std::vector<int>& input_lengths,
                      std::vector<int>& input_offsets,
                      std::vector<std::vector<int>>& output_lengths_all,
                      std::vector<std::vector<int>>& output_offsets_all) {
    assert(this->isInitialized == true);
    NetWrapper::init(rank, nranks, input, output, input_size, output_size);
    assert(input.ptr != output.ptr);

    // update stream, but actually in high level, torch level, we hope to reuse the same stream, like pytorch nccl
    // plugin does
    this->setStream(stream);

    // update the split way
    initAlltoallv(input_lengths, input_offsets, output_lengths_all, output_offsets_all);

    // if pytorch could fix one tensor's physical memory, then we don't need to copy the input and output tensor
    // is this necessary to add a new kernel to copy the input and output tensor?
    CUDA_CHECK(cudaMemcpyAsync(
        this->pllm_tensor_input.ptr, input_tensor_ptr, input_size * sizeof(Element), cudaMemcpyDeviceToDevice, stream));
    //   CUDA_CHECK(cudaMemcpyAsync(this->pllm_tensor_output.ptr,
    //                              output_tensor_ptr,
    //                              output_size * sizeof(Element),
    //                              cudaMemcpyDeviceToDevice,
    //                              stream));
  }

  void update_wrapper_without_copy(int rank,
                                  int nranks,
                                  void* input_tensor_ptr,
                                  void* output_tensor_ptr,
                                  size_t input_size,
                                  size_t output_size,
                                  cudaStream_t stream,
                                  std::vector<int>& input_lengths,
                                  std::vector<int>& input_offsets,
                                  std::vector<std::vector<int>>& output_lengths_all,
                                  std::vector<std::vector<int>>& output_offsets_all) {
    assert(this->isInitialized == true);
    NetWrapper::init(rank, nranks, input, output, input_size, output_size);
    assert(input.ptr != output.ptr);

    // update stream, but actually in high level, torch level, we hope to reuse the same stream, like pytorch nccl
    // plugin does
    this->setStream(stream);

    // update the split way
    initAlltoallv(input_lengths, input_offsets, output_lengths_all, output_offsets_all);

    this->tmp_input_tensor_ptr = (Element *)input_tensor_ptr;
    this->tmp_output_tensor_ptr = (Element *)output_tensor_ptr;
    this->fuse_copy_mode = true;

    // if pytorch could fix one tensor's physical memory, then we don't need to copy the input and output tensor
    // is this necessary to add a new kernel to copy the input and output tensor?
    // CUDA_CHECK(cudaMemcpyAsync(
    // this->pllm_tensor_input.ptr, input_tensor_ptr, input_size * sizeof(Element), cudaMemcpyDeviceToDevice, stream));
    //   CUDA_CHECK(cudaMemcpyAsync(this->pllm_tensor_output.ptr,
    //                              output_tensor_ptr,
    //                              output_size * sizeof(Element),
    //                              cudaMemcpyDeviceToDevice,
    //                              stream));
  }

  int rank_idx(int rank, std::vector<int>& ranks) {
    for (int i = 0; i < ranks.size(); i++) {
      if (rank == ranks[i]) {
        return i;
      }
    }
    return -1;
  }

  ~NetAlltoAllv() {
    CUDA_CHECK(cudaFreeAsync(syncersCuda, stream));
    CUDA_CHECK(cudaFreeAsync(smChannelHandlesCuda, stream));
    if (input_lengths != nullptr) {
      CUDA_CHECK(cudaFreeAsync(input_lengths, stream));
    }
    // CUDA_CHECK(cudaFreeAsync(pllm_tensor_input.ptr, stream));
    // CUDA_CHECK(cudaFreeAsync(pllm_tensor_output.ptr, stream));
  }

  NetAlltoAllv& setColumnwise(bool columnwise = true) {
    this->columnwise = columnwise;
    return *this;
  }

  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) {
    // why not true?
    assert(sync == True);
    assert(columnwise == False);

    const int nchannels = nranks - 1;
    if (!columnwise) {
      // assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      // assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2);
      // const uint64_t nelem_per_shard = input_size / nranks;
      // const int64_t local_offset = this->rank_idx_in_send[rank] * nelem_per_shard;
      if (sync) {
        if (this->fuse_copy_mode) {
          assert(this->tmp_input_tensor_ptr != nullptr);
          assert(this->tmp_output_tensor_ptr != nullptr);
          alltoallvFuseCopyKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(smChannelHandlesCuda,
                                                                              syncersCuda,
                                                                              rank,
                                                                              nchannels,
                                                                              input_lengths,
                                                                              input_offsets,
                                                                              output_lengths_all,
                                                                              output_offsets_all,
                                                                              input,
                                                                              output,
                                                                              tmp_input_tensor_ptr,
                                                                              tmp_output_tensor_ptr);
          this->tmp_input_tensor_ptr = nullptr;
          this->tmp_output_tensor_ptr = nullptr;
        }
        else {
          alltoallvKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(smChannelHandlesCuda,
                                                                      syncersCuda,
                                                                      rank,
                                                                      nchannels,
                                                                      input_lengths,
                                                                      input_offsets,
                                                                      output_lengths_all,
                                                                      output_offsets_all,
                                                                      input,
                                                                      output);

        }
      } else {
        // alltoallKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
        //     smChannelHandlesCuda, syncersCuda, nchannels, local_offset, nelem_per_shard_, input, output);
      }
    } else {
      // assert(this->pllm_tensor_input.dim1 == this->pllm_tensor_output.dim1);
      // assert(this->pllm_tensor_input.dim2 == this->pllm_tensor_output.dim2 / nranks);
      // const uint64_t input_ncols = this->pllm_tensor_input.dim2;
      // const uint64_t output_ncols = this->pllm_tensor_output.dim2;
      // const uint64_t output_row_offset = rank * input_ncols;
      // const uint64_t nrows = this->pllm_tensor_input.dim1;

      // constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(cutlass::half_t);
      // assert(input_ncols % n_half_per_int4 == 0);
      // assert(output_ncols % n_half_per_int4 == 0);
      // assert(output_row_offset % n_half_per_int4 == 0);
      // assert(reinterpret_cast<uintptr_t>(input) % sizeof(int4) == 0);
      // assert(reinterpret_cast<uintptr_t>(output) % sizeof(int4) == 0);

      // columnwiseAllgatherKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
      //     smChannelHandlesCuda, syncersCuda, sync, nchannels,
      //     input_ncols, output_ncols, output_row_offset, nrows, input, output);
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smChannelHandlesCuda, nchannels);
  }

  // override copy constructor
  NetAlltoAllv(const NetAllGather& other) = delete;
};

class NetReduceScatter : public NetWrapper {
  public:
  mscclpp::DeviceSyncer* syncerCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smInputBuffChannelHandlesCuda;

  NetReduceScatter() : NetWrapper() {}
  ~NetReduceScatter() {
    CUDA_CHECK(cudaFree(syncerCuda));
    CUDA_CHECK(cudaFree(smInputBuffChannelHandlesCuda));
  }
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            Element* input,
            Element* output,
            size_t input_size,
            size_t output_size) {
    NetWrapper::init(rank, nranks, input, output, input_size, output_size);

    setupSmChannels(comm, connections, &smInputBuffChannelHandlesCuda, input, input_size);
    mscclpp::DeviceSyncer syncer = mscclpp::DeviceSyncer();
    CUDA_CHECK(cudaMalloc(&syncerCuda, sizeof(mscclpp::DeviceSyncer)));
    CUDA_CHECK(cudaMemcpy(syncerCuda, &syncer, sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));
  }
  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
    const uint64_t nelem_per_shard = input_size / nranks;
    if (sync) {
      reduceScatterKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
          smInputBuffChannelHandlesCuda, syncerCuda, rank, nranks, nelem_per_shard, input, output);
    } else {
      reduceScatterKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
          smInputBuffChannelHandlesCuda, syncerCuda, rank, nranks, nelem_per_shard, input, output);
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smInputBuffChannelHandlesCuda, nchannels);
  }
};

class NetAllReduce : public NetWrapper {
  public:
  mscclpp::DeviceSyncer* syncersCuda;
  mscclpp::DeviceSyncer* globalSyncerCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smInputBuffChannelHandlesCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutputBuffChannelHandlesCuda;
  Element* tmp_input_tensor_ptr = nullptr;
  Element* tmp_output_tensor_ptr = nullptr;
  bool fuse_copy_mode = false;
  bool bf16_mode = false;

  NetAllReduce() : NetWrapper() {}

  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,
            pllmTensor<Element> output) {
    assert(input.layout == output.layout);
    this->pllm_tensor_input = input;
    this->pllm_tensor_output = output;
    NetWrapper::init(rank, nranks, input.ptr, output.ptr, input.size(), output.size());
    size_t elem_bytes = bf16_mode ? sizeof(cutlass::bfloat16_t) : sizeof(Element);
    setupSmChannelsBytes(comm, connections, &smInputBuffChannelHandlesCuda, input.ptr, input.ptr, input.size(), output.size(), elem_bytes);
    setupSmChannelsBytes(comm, connections, &smOutputBuffChannelHandlesCuda, output.ptr, output.ptr, output.size(), output.size(), elem_bytes);

    std::vector<mscclpp::DeviceSyncer> syncers(nranks - 1);
    CUDA_CHECK(cudaMalloc(&syncersCuda, syncers.size() * sizeof(mscclpp::DeviceSyncer)));
    CUDA_CHECK(cudaMemcpy(
        syncersCuda, syncers.data(), syncers.size() * sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));

    mscclpp::DeviceSyncer syncer = mscclpp::DeviceSyncer();
    CUDA_CHECK(cudaMalloc(&globalSyncerCuda, sizeof(mscclpp::DeviceSyncer)));
    CUDA_CHECK(cudaMemcpy(globalSyncerCuda, &syncer, sizeof(mscclpp::DeviceSyncer), cudaMemcpyHostToDevice));
  }

  ~NetAllReduce() {
    CUDA_CHECK(cudaFree(syncersCuda));
    CUDA_CHECK(cudaFree(globalSyncerCuda));
    CUDA_CHECK(cudaFree(smInputBuffChannelHandlesCuda));
    CUDA_CHECK(cudaFree(smOutputBuffChannelHandlesCuda));
  }
  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
    const int nchannels = nranks - 1;
    const uint64_t nelem_per_shard = input_size / nranks;
    const uint64_t local_offset = rank * nelem_per_shard;
    if (sync) {
      if (this->fuse_copy_mode) {
        if (bf16_mode) {
          allreduceFuseCopyKernelEntryPointBF16<<<nblocks, nthreads, 0, stream>>>(
              smInputBuffChannelHandlesCuda,
              smOutputBuffChannelHandlesCuda,
              syncersCuda,
              globalSyncerCuda,
              nchannels,
              local_offset,
              rank,
              nranks,
              nelem_per_shard,
              reinterpret_cast<cutlass::bfloat16_t*>(input),
              reinterpret_cast<cutlass::bfloat16_t*>(output),
              reinterpret_cast<cutlass::bfloat16_t*>(this->tmp_input_tensor_ptr),
              reinterpret_cast<cutlass::bfloat16_t*>(this->tmp_output_tensor_ptr));
        } else {
          allreduceFuseCopyKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
              smInputBuffChannelHandlesCuda,
              smOutputBuffChannelHandlesCuda,
              syncersCuda,
              globalSyncerCuda,
              nchannels,
              local_offset,
              rank,
              nranks,
              nelem_per_shard,
              input,
              output,
              this->tmp_input_tensor_ptr,
              this->tmp_output_tensor_ptr);
        }
      } else {
        if (bf16_mode) {
          allreduceKernelEntryPointBF16<<<nblocks, nthreads, 0, stream>>>(
              smInputBuffChannelHandlesCuda,
              smOutputBuffChannelHandlesCuda,
              syncersCuda,
              globalSyncerCuda,
              nchannels,
              local_offset,
              rank,
              nranks,
              nelem_per_shard,
              reinterpret_cast<cutlass::bfloat16_t*>(input),
              reinterpret_cast<cutlass::bfloat16_t*>(output));
        } else {
          allreduceKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(
              smInputBuffChannelHandlesCuda,
              smOutputBuffChannelHandlesCuda,
              syncersCuda,
              globalSyncerCuda,
              nchannels,
              local_offset,
              rank,
              nranks,
              nelem_per_shard,
              input,
              output);
        }
      }
    } else {
      if (bf16_mode) {
        allreduceKernelWithoutSyncBF16<<<nblocks, nthreads, 0, stream>>>(
            smInputBuffChannelHandlesCuda,
            smOutputBuffChannelHandlesCuda,
            syncersCuda,
            globalSyncerCuda,
            nchannels,
            local_offset,
            rank,
            nranks,
            nelem_per_shard,
            reinterpret_cast<cutlass::bfloat16_t*>(input),
            reinterpret_cast<cutlass::bfloat16_t*>(output));
      } else {
        allreduceKernelWithoutSync<<<nblocks, nthreads, 0, stream>>>(
            smInputBuffChannelHandlesCuda,
            smOutputBuffChannelHandlesCuda,
            syncersCuda,
            globalSyncerCuda,
            nchannels,
            local_offset,
            rank,
            nranks,
            nelem_per_shard,
            input,
            output);
      }
    }
  }
  void sync(cudaStream_t stream) override {
    const int nchannels = nranks - 1;
    syncDevices<<<1, nchannels, 0, stream>>>(smInputBuffChannelHandlesCuda, nchannels);
  }
  void update_wrapper_without_copy(int rank,
                                  int nranks,
                                  void* input_tensor_ptr,
                                  void* output_tensor_ptr,
                                  size_t input_size,
                                  size_t output_size,
                                  cudaStream_t stream,
                                  int sm_num,
                                  int block_size) {
    assert(this->isInitialized == true);
    NetWrapper::init(rank, nranks, input, output, input_size, output_size);
    assert(input.ptr != output.ptr);

    // update stream, but actually in high level, torch level, we hope to reuse the same stream, like pytorch nccl
    // plugin does
    this->setStream(stream);
    this->configRun(sm_num, block_size, true);
    // this->current_comm_ranks = group_ranks;
    this->tmp_input_tensor_ptr = (Element*)input_tensor_ptr;
    this->tmp_output_tensor_ptr = (Element*)output_tensor_ptr;
    this->fuse_copy_mode = true;
  }
};

class NetAllReduceWithLN : public NetAllReduce {
  public:
  pllmTensor<half> ln_weight;
  float epsilon;
  bool run_ln = true;
  pllmTensor<Element> output_before_ln;

  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            pllmTensor<Element> input,
            pllmTensor<Element> output,
            pllmTensor<Element> output_before_ln) {
    NetAllReduce::init(comm, connections, rank, nranks, input, output);
    this->output_before_ln = output_before_ln;
  }

  void setEpsilon(float epsilon) {
    this->epsilon = epsilon;
  }

  // bool setWeight(vortexWeight weight) {
  //     ln_weight = pllmTensor<half>(weight.ptr, weight.size());
  //     return true;
  // }

  NetAllReduceWithLN& runLn(bool run_ln = true) {
    this->run_ln = run_ln;
    return *this;
  }

  void operator()(cudaStream_t stream, int nblocks, int nthreads, bool sync) override {
    const int nchannels = nranks - 1;
    const uint64_t nelem_per_shard = input_size / nranks;
    const uint64_t local_offset = rank * nelem_per_shard;
    if (run_ln) {
      // spdlog::info("row {}, col {}", pllm_tensor_input.dim1/nranks, pllm_tensor_input.dim2);
      allreduceKernelWithLNEntryPoint<<<nblocks, nthreads, 0, stream>>>(smInputBuffChannelHandlesCuda,
                                                                        smOutputBuffChannelHandlesCuda,
                                                                        syncersCuda,
                                                                        globalSyncerCuda,
                                                                        nchannels,
                                                                        local_offset,
                                                                        rank,
                                                                        nranks,
                                                                        nelem_per_shard,
                                                                        input,
                                                                        output,
                                                                        output_before_ln.ptr,
                                                                        ln_weight.ptr,
                                                                        pllm_tensor_input.dim1 / nranks,
                                                                        pllm_tensor_input.dim2,
                                                                        epsilon);
    } else {
      allreduceKernelEntryPoint<<<nblocks, nthreads, 0, stream>>>(smInputBuffChannelHandlesCuda,
                                                                  smOutputBuffChannelHandlesCuda,
                                                                  syncersCuda,
                                                                  globalSyncerCuda,
                                                                  nchannels,
                                                                  local_offset,
                                                                  rank,
                                                                  nranks,
                                                                  nelem_per_shard,
                                                                  input,
                                                                  output);
    }
  }

  OperatorWrapper& logImpl(std::shared_ptr<spdlog::logger> logger = default_logger) override {
    log_tensor(logger, name + "input", getInput(), 10, 20);
    log_tensor(logger, name + "output", getOutput(), 10, 20);
    log_tensor(logger, name + "output_before_ln", output_before_ln, 10, 20);
    log_tensor(logger, name + "ln_weight", ln_weight, 1, 20);
    return *this;
  }
};

class NetAsyncWrapper : public NetWrapper {
  public:
  std::vector<Element*> remoteInputBuffs;
  std::vector<mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>> proxyChannelHandles;

  NetAsyncWrapper() {}
  void init(int rank, int nranks, Element* input, Element* output, int input_size, int output_size) {
    NetWrapper::init(rank, nranks, input, output, input_size, output_size);
    remoteInputBuffs = getRemoteBuff(input);
    assert(remoteInputBuffs.size() == nranks - 1);
  }
  ~NetAsyncWrapper() {
    for (size_t i = 0; i < remoteInputBuffs.size(); ++i) {
#ifdef ENABLE_MPI
      CUDA_CHECK(cudaIpcCloseMemHandle(remoteInputBuffs[i]));
#endif
    }
  }
  virtual void start(cudaStream_t stream, int nblocks = 1, int nthreads = 8) {}
  virtual void finish(cudaStream_t stream, int nblocks = 8, int nthreads = 1024) {}

  protected:
  std::vector<Element*> getRemoteBuff(Element* localBuff) {
    std::vector<Element*> remoteBuffs;
    for (size_t r = 0; r < nranks; ++r) {
#ifdef ENABLE_MPI
      cudaIpcMemHandle_t handle;
#else
      static Element* globalBuff;
#endif
      if (r == rank) {
#ifdef ENABLE_MPI
        CUDA_CHECK(cudaIpcGetMemHandle(&handle, localBuff));
        MPI_Bcast(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, r, MPI_COMM_WORLD);
#else
        globalBuff = localBuff;
        worker_sync->barrier();  // wait for all peers to receive handle
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        std::cout << "rank " << rank << " setup handles on cuda dev " << device << std::endl;
        worker_sync->barrier();  // wait for all peers to done with the handle
#endif
      } else {
#ifdef ENABLE_MPI
        Element* remoteBuff;
        MPI_Bcast(&handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, r, MPI_COMM_WORLD);
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&remoteBuff, handle, cudaIpcMemLazyEnablePeerAccess));
#else
        worker_sync->barrier();
        Element* remoteBuff = globalBuff;
        worker_sync->barrier();
#endif
        remoteBuffs.push_back(remoteBuff);
      }
    }
    return remoteBuffs;
  }
  void setupProxyChannels(std::shared_ptr<mscclpp::ProxyService> service,
                          std::shared_ptr<mscclpp::Communicator> comm,
                          std::vector<std::shared_ptr<mscclpp::Connection>> connections,
                          mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>** proxyChannelHandlesCuda,
                          Element* input,
                          Element* output,
                          int input_size,
                          int output_size) {
    const mscclpp::TransportFlags allTransports = mscclpp::Transport::CudaIpc;
    mscclpp::RegisteredMemory inputBuffRegMem =
        comm->registerMemory(input, input_size * sizeof(Element), allTransports);
    mscclpp::RegisteredMemory outputBuffRegMem;
    if (input != output)
      outputBuffRegMem = comm->registerMemory(output, output_size * sizeof(Element), allTransports);

    std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
    mscclpp::RegisteredMemory& localRegMemory = (input != output) ? outputBuffRegMem : inputBuffRegMem;

    for (size_t r = 0; r < nranks; ++r) {
      if (r == rank)
        continue;
      comm->sendMemoryOnSetup(localRegMemory, r, 0);
      auto remoteMemory = comm->recvMemoryOnSetup(r, 0);
      remoteRegMemories.push_back(remoteMemory);
    }
    comm->setup();
    for (size_t i = 0; i < connections.size(); ++i) {
      proxyChannelHandles.push_back(mscclpp::deviceHandle(
          mscclpp::SimpleProxyChannel(service->proxyChannel(service->buildAndAddSemaphore(*comm, connections[i])),
                                      service->addMemory(remoteRegMemories[i].get()),
                                      service->addMemory(inputBuffRegMem))));
    }
    comm->setup();

    assert(connections.size() == nranks - 1);
    CUDA_CHECK(
        cudaMalloc(proxyChannelHandlesCuda, (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>)));
    CUDA_CHECK(cudaMemcpy(*proxyChannelHandlesCuda,
                          &proxyChannelHandles[proxyChannelHandles.size() - (nranks - 1)],
                          (nranks - 1) * sizeof(mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel>),
                          cudaMemcpyHostToDevice));
  }
};

class NetAllGatherAsync : public NetAsyncWrapper {
  public:
  cudaEvent_t mainEvent;
  std::vector<cudaEvent_t> memcpyEvents;
  std::vector<cudaStream_t> memcpyStreams;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smSyncChannelHandlesCuda;

  NetAllGatherAsync() : NetAsyncWrapper() {}
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            Element* input,
            Element* output,
            int input_size,
            int output_size) {
    NetAsyncWrapper::init(rank, nranks, input, output, input_size, output_size);
    setupSmChannels(comm, connections, &smSyncChannelHandlesCuda, input, input_size);

    CUDA_CHECK(cudaEventCreate(&mainEvent));
    memcpyEvents.resize(nranks);
    memcpyStreams.resize(nranks);
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaEventCreate(&memcpyEvents[r]));
      CUDA_CHECK(cudaStreamCreate(&memcpyStreams[r]));
    }
  }
  ~NetAllGatherAsync() {
    CUDA_CHECK(cudaFree(smSyncChannelHandlesCuda));
    CUDA_CHECK(cudaEventDestroy(mainEvent));
    for (size_t i = 0; i < nranks; ++i) {
      CUDA_CHECK(cudaEventDestroy(memcpyEvents[i]));
      CUDA_CHECK(cudaStreamDestroy(memcpyStreams[i]));
    }
  }
  void start(cudaStream_t stream, int nblocks = 1, int nthreads = 8) override {
    const int nchannels = nranks - 1;
    assert(nchannels <= nblocks * nthreads);
    const uint64_t nelem_per_shard = input_size / nranks;
    const uint64_t local_offset = rank * nelem_per_shard;
    syncDevices<<<1, nchannels, 0, stream>>>(smSyncChannelHandlesCuda, nchannels);
    cudaEventRecord(mainEvent, stream);
    for (size_t r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaStreamWaitEvent(memcpyStreams[r], mainEvent, 0));
      if (r != rank) {
        CUDA_CHECK(cudaMemcpyAsync(output + r * nelem_per_shard,
                                   remoteInputBuffs[r > rank ? r - 1 : r] + r * nelem_per_shard,
                                   nelem_per_shard * sizeof(Element),
                                   cudaMemcpyDeviceToDevice,
                                   memcpyStreams[r > rank ? r - 1 : r]));
      } else if (input != output) {
        CUDA_CHECK(cudaMemcpyAsync(output + local_offset,
                                   input + local_offset,
                                   nelem_per_shard * sizeof(Element),
                                   cudaMemcpyDeviceToDevice,
                                   memcpyStreams[r]));
      }
      CUDA_CHECK(cudaEventRecord(memcpyEvents[r], memcpyStreams[r]));
    }
  }
  void finish(cudaStream_t stream, int nblocks = 1, int nthreads = 8) override {
    const int nchannels = nranks - 1;
    assert(nchannels <= nblocks * nthreads);
    for (size_t i = 0; i < nranks; ++i) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, memcpyEvents[i], 0));
    }
    syncDevices<<<1, nchannels, 0, stream>>>(smSyncChannelHandlesCuda, nchannels);
  }
};

class NetReduceScatterAsync : public NetAsyncWrapper {
  public:
  static constexpr uint64_t n_half_per_int4 = sizeof(int4) / sizeof(Element);
  cudaEvent_t mainEvent;
  std::vector<cudaEvent_t> memcpyEvents;
  std::vector<cudaStream_t> memcpyStreams;
  std::vector<Element*> scratches;
  Element** scratchesCuda;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smSyncChannelHandlesCuda;

  NetReduceScatterAsync() : NetAsyncWrapper() {}
  void init(std::shared_ptr<mscclpp::Communicator> comm,
            std::vector<std::shared_ptr<mscclpp::Connection>> connections,
            int rank,
            int nranks,
            Element* input,
            Element* output,
            int input_size,
            int output_size) {
    NetAsyncWrapper::init(rank, nranks, input, output, input_size, output_size);

    setupSmChannels(comm, connections, &smSyncChannelHandlesCuda, input, input_size);

    const uint64_t nelem_per_shard = input_size / nranks;
    scratches.resize(nranks - 1);
    for (int i = 0; i < nranks - 1; ++i) {
      CUDA_CHECK(cudaMalloc(&scratches[i], (nelem_per_shard + n_half_per_int4) * sizeof(Element)));
    }
    CUDA_CHECK(cudaMalloc(&scratchesCuda, scratches.size() * sizeof(Element*)));
    CUDA_CHECK(
        cudaMemcpy(scratchesCuda, scratches.data(), scratches.size() * sizeof(Element*), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&mainEvent));
    memcpyEvents.resize(nranks - 1);
    memcpyStreams.resize(nranks - 1);
    for (int i = 0; i < nranks - 1; ++i) {
      CUDA_CHECK(cudaEventCreate(&memcpyEvents[i]));
      CUDA_CHECK(cudaStreamCreate(&memcpyStreams[i]));
    }
  }
  ~NetReduceScatterAsync() {
    for (size_t i = 0; i < nranks - 1; ++i)
      CUDA_CHECK(cudaFree(scratches[i]));
    CUDA_CHECK(cudaFree(scratchesCuda));
    CUDA_CHECK(cudaFree(smSyncChannelHandlesCuda));
    CUDA_CHECK(cudaEventDestroy(mainEvent));
    for (size_t i = 0; i < nranks - 1; ++i) {
      CUDA_CHECK(cudaEventDestroy(memcpyEvents[i]));
      CUDA_CHECK(cudaStreamDestroy(memcpyStreams[i]));
    }
  }
  void start(cudaStream_t stream, int nblocks = 1, int nthreads = 8) override {
    assert(nranks - 1 <= nblocks * nthreads);
    const uint64_t nelem_per_shard = input_size / nranks;
    const uint64_t offset = rank * nelem_per_shard;
    const uint64_t offset4 = (offset + n_half_per_int4 - 1) / n_half_per_int4;
    const uint64_t nFirstElem = offset4 * n_half_per_int4 - offset;
    const uint64_t scratch_start = n_half_per_int4 - nFirstElem;
    syncDevices<<<1, nranks - 1, 0, stream>>>(smSyncChannelHandlesCuda, nranks - 1);
    cudaEventRecord(mainEvent, stream);
    for (size_t i = 0; i < nranks - 1; ++i) {
      CUDA_CHECK(cudaStreamWaitEvent(memcpyStreams[i], mainEvent, 0));
      CUDA_CHECK(cudaMemcpyAsync(scratches[i] + scratch_start,
                                 remoteInputBuffs[i] + nelem_per_shard * rank,
                                 nelem_per_shard * sizeof(Element),
                                 cudaMemcpyDeviceToDevice,
                                 memcpyStreams[i]));
      CUDA_CHECK(cudaEventRecord(memcpyEvents[i], memcpyStreams[i]));
    }
  }
  void finish(cudaStream_t stream, int nblocks = 8, int nthreads = 1024) override {
    assert(nranks - 1 <= nblocks * nthreads);
    const uint64_t nelem_per_shard = input_size / nranks;
    for (size_t i = 0; i < nranks - 1; ++i) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, memcpyEvents[i], 0));
    }
    asyncReduceKernel<<<nblocks, nthreads, 0, stream>>>(rank, nranks, nelem_per_shard, scratchesCuda, input, output);
    syncDevices<<<1, nranks - 1, 0, stream>>>(smSyncChannelHandlesCuda, nranks - 1);
  }
};
