// some helper functions of mscclpp, for python binding
#include <algorithm>
#include <cmath>
#include <random>

#include "comm_wrapper.cuh"
#include "cutlassGemmWrapper.cuh"
#include "netWrapper.cuh"
#include "spdlog/spdlog.h"
#include "tensorLogger.cuh"

std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
// Add a global connections map to store connections per communicator
std::unordered_map<std::shared_ptr<mscclpp::Communicator>, std::vector<std::shared_ptr<mscclpp::Connection>>>
    connection_cache;

/*****************************************************/
/*
    init mscclpp communicator, this is done once in the beginning
*/
/*****************************************************/

// init mscclpp communicator, this is done once in the beginning
void init_bootstrap(int rank, int nranks) {
  bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
}

mscclpp::UniqueId get_unique_id() {
  return bootstrap->createUniqueId();
}

std::vector<std::shared_ptr<mscclpp::Connection>>& init_connections(std::shared_ptr<mscclpp::Communicator> comm,
                                                                    int rank,
                                                                    int nranks) {
  // Check if connections already exist for this communicator
  auto it = connection_cache.find(comm);
  if (it != connection_cache.end()) {
    return it->second;
  }

  // Initialize new connections if not found
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
  for (int r = 0; r < nranks; ++r) {
    if (r == rank)
      continue;
    mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
    connectionFutures.push_back(comm->connectOnSetup(r, 0, transport));
  }
  comm->setup();
  std::transform(
      connectionFutures.begin(),
      connectionFutures.end(),
      std::back_inserter(connections),
      [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });

  // Store in cache and return
  connection_cache[comm] = connections;
  return connection_cache[comm];
}

// std::shared_ptr<mscclpp::Communicator> mscclppCommInitRank(int world_size, mscclpp::UniqueId uniqueId, int rank)
std::shared_ptr<CommunicatorWrapper> mscclppCommInitRank(int world_size, mscclpp::UniqueId uniqueId, int rank) {
  bootstrap->initialize(uniqueId);
  auto mscclpp_comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  auto& connections = init_connections(mscclpp_comm, rank, world_size);
  auto comm = std::make_shared<CommunicatorWrapper>(mscclpp_comm, connections, rank, world_size);
  return comm;
}

/*****************************************************/
/*
    cached custom functions, for python binding, to remove high cpu/cudaIpc sync overhead
*/
/*****************************************************/

void cached_Custom_MScclpp_AlltoAllv(std::shared_ptr<CommunicatorWrapper> comm,
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
                                     std::vector<std::vector<int>>& output_offsets_all) {
  assert(comm->wrapper_sptr != nullptr);
  std::shared_ptr<NetAlltoAllv>& wrapper_ptr = comm->wrapper_sptr;
  wrapper_ptr->setStream(stream);
  wrapper_ptr->update_wrapper(comm->rank,
                              comm->nranks,
                              input_buff,
                              output_buff,
                              input_size,
                              output_size,
                              stream,
                              input_lengths,
                              input_offsets,
                              output_lengths_all,
                              output_offsets_all);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
  // CUDA_CHECK(cudaMemcpyAsync(input_buff,
  //                            wrapper_ptr->pllm_tensor_input.ptr,
  //                            input_size * sizeof(cutlass::half_t),
  //                            cudaMemcpyDeviceToDevice,
  //                            stream));
  CUDA_CHECK(cudaMemcpyAsync(output_buff,
                             wrapper_ptr->pllm_tensor_output.ptr,
                             output_size * sizeof(cutlass::half_t),
                             cudaMemcpyDeviceToDevice,
                             stream));
  // CUDA_CHECK(cudaDeviceSynchronize());
}

void cached_Custom_MScclpp_AlltoAllv_FuseCopy(std::shared_ptr<CommunicatorWrapper> comm,
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
                                              std::vector<std::vector<int>>& output_offsets_all) {
  assert(comm->wrapper_sptr != nullptr);
  std::shared_ptr<NetAlltoAllv>& wrapper_ptr = comm->wrapper_sptr;
  wrapper_ptr->setStream(stream);
  wrapper_ptr->update_wrapper_without_copy(comm->rank,
                                          comm->nranks,
                                          input_buff,
                                          output_buff,
                                          input_size,
                                          output_size,
                                          stream,
                                          input_lengths,
                                          input_offsets,
                                          output_lengths_all,
                                          output_offsets_all);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
  // CUDA_CHECK(cudaMemcpyAsync(input_buff,
  //                            wrapper_ptr->pllm_tensor_input.ptr,
  //                            input_size * sizeof(cutlass::half_t),
  //                            cudaMemcpyDeviceToDevice,
  //                            stream));
  // CUDA_CHECK(cudaMemcpyAsync(output_buff,
  //                           wrapper_ptr->pllm_tensor_output.ptr,
  //                           output_size * sizeof(cutlass::half_t),
  //                           cudaMemcpyDeviceToDevice,
  //                           stream));
  // CUDA_CHECK(cudaDeviceSynchronize());
}

void Custom_MScclpp_AllReduce(std::shared_ptr<CommunicatorWrapper> comm,
                              cudaStream_t stream,
                              int sm_num,
                              int block_size) {
  assert(comm->ar_wrapper_sptr != nullptr);
  std::shared_ptr<NetAllReduce>& wrapper_ptr = comm->ar_wrapper_sptr;
  wrapper_ptr->fuse_copy_mode = false;
  wrapper_ptr->setStream(stream);
  wrapper_ptr->configRun(sm_num, block_size, true);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
}

void Custom_MScclpp_AllReduce_BF16(std::shared_ptr<CommunicatorWrapper> comm,
                                   cudaStream_t stream,
                                   int sm_num,
                                   int block_size) {
  assert(comm->ar_wrapper_sptr != nullptr);
  std::shared_ptr<NetAllReduce>& wrapper_ptr = comm->ar_wrapper_sptr;
  wrapper_ptr->bf16_mode = true;
  wrapper_ptr->fuse_copy_mode = false;
  wrapper_ptr->setStream(stream);
  wrapper_ptr->configRun(sm_num, block_size, true);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
}

void Custom_MScclpp_AllReduce_FuseCopy(std::shared_ptr<CommunicatorWrapper> comm,
                                      void* input_buff,
                                      void* output_buff,
                                      const size_t input_size,
                                      const size_t output_size,
                                      cudaStream_t stream,
                                      int sm_num,
                                      int block_size) {
  assert(comm->ar_wrapper_sptr != nullptr);
  std::shared_ptr<NetAllReduce>& wrapper_ptr = comm->ar_wrapper_sptr;
  // wrapper_ptr->setStream(stream);
  // wrapper_ptr->configRun(sm_num, block_size, true);
  wrapper_ptr->update_wrapper_without_copy(comm->rank,
                                          comm->nranks,
                                          input_buff,
                                          output_buff,
                                          input_size,
                                          output_size,
                                          stream,
                                          sm_num,
                                          block_size);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
}

void Custom_MScclpp_AllReduce_FuseCopy_BF16(std::shared_ptr<CommunicatorWrapper> comm,
                                            void* input_buff,
                                            void* output_buff,
                                            const size_t input_size,
                                            const size_t output_size,
                                            cudaStream_t stream,
                                            int sm_num,
                                            int block_size) {
  assert(comm->ar_wrapper_sptr != nullptr);
  std::shared_ptr<NetAllReduce>& wrapper_ptr = comm->ar_wrapper_sptr;
  wrapper_ptr->bf16_mode = true;
  wrapper_ptr->update_wrapper_without_copy(comm->rank,
                                           comm->nranks,
                                           input_buff,
                                           output_buff,
                                           input_size,
                                           output_size,
                                           stream,
                                           sm_num,
                                           block_size);
  wrapper_ptr->operator()(wrapper_ptr->stream, wrapper_ptr->nblocks, wrapper_ptr->nthreads, wrapper_ptr->sync_mode);
}

/*
    below are deprecated functions, but for performance testing, we would like to keep them.
*/

/*****************************************************/
/*
    init net wrapper, but the impl here has high cpu/cudaIpc sync overhead
*/
/*****************************************************/

void init_NetAlltoAll(NetAlltoAll& wrapper,
                      std::shared_ptr<CommunicatorWrapper> comm,
                      std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                      const int rank,
                      const int nranks,
                      void* input_buff,
                      void* output_buff,
                      const size_t buff_size,
                      cudaStream_t stream,
                      int sm_num,
                      int block_size) {
  // // Check alignment
  // constexpr size_t HALF_ALIGNMENT = alignof(cutlass::half_t);
  // if ((reinterpret_cast<std::uintptr_t>(input_buff) % HALF_ALIGNMENT) != 0 ||
  //     (reinterpret_cast<std::uintptr_t>(output_buff) % HALF_ALIGNMENT) != 0) {
  //     throw std::runtime_error("Input or output buffer is not properly aligned for half precision");
  // }

  int dim1, input_dim2, output_dim2;
  dim1 = buff_size / sizeof(cutlass::half_t);
  input_dim2 = 1;
  output_dim2 = 1;

  bool sync = true;
  wrapper.setStream(stream);
  wrapper.configRun(sm_num, block_size, sync);
  wrapper.init(comm->comm,
               connections,
               rank,
               nranks,
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, dim1, output_dim2, PllmLayout::ROW_MAJOR});
}

void init_NetAlltoAllUneven(NetAlltoAllUneven& wrapper,
                            std::shared_ptr<CommunicatorWrapper> comm,
                            std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                            const int rank,
                            const int nranks,
                            void* input_buff,
                            void* output_buff,
                            const size_t buff_size,
                            cudaStream_t stream,
                            int sm_num,
                            int block_size,
                            std::vector<int>& ranks_send,
                            std::vector<int>& ranks_recv) {
  int dim1, input_dim2, output_dim2;
  dim1 = buff_size / sizeof(cutlass::half_t);
  input_dim2 = 1;
  output_dim2 = 1;
  size_t output_buff_size = buff_size * ranks_send.size() / ranks_recv.size();
  int output_dim = output_buff_size / sizeof(cutlass::half_t);

  bool sync = true;  // async doesn't support for now
  wrapper.setStream(stream);
  wrapper.configRun(sm_num, block_size, sync);
  wrapper.init(
      comm->comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, output_dim, output_dim2, PllmLayout::ROW_MAJOR},
      ranks_send,
      ranks_recv);
}

void init_NetAlltoAllv(NetAlltoAllv& wrapper,
                       std::shared_ptr<CommunicatorWrapper> comm,
                       std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                       const int rank,
                       const int nranks,
                       void* input_buff,
                       void* output_buff,
                       const size_t input_size,   // input_buff_size = input_size * sizeof(cutlass::half_t)
                       const size_t output_size,  // output_buff_size = output_size * sizeof(cutlass::half_t)
                       cudaStream_t stream,
                       int sm_num,
                       int block_size,
                       std::vector<int>& input_lengths,
                       std::vector<int>& input_offsets,
                       std::vector<std::vector<int>>& output_lengths_all,
                       std::vector<std::vector<int>>& output_offsets_all) {
  bool sync = true;
  wrapper.setStream(stream);
  wrapper.configRun(sm_num, block_size, sync);
  wrapper.init(
      comm->comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, input_size, (size_t)1, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, output_size, (size_t)1, PllmLayout::ROW_MAJOR},
      input_lengths,
      input_offsets,
      output_lengths_all,
      output_offsets_all);
}

/*****************************************************/
/*
    custom functions, for python binding, but still high cpu/cudaIpc sync overhead
*/
/*****************************************************/

void Custom_MScclpp_UnevenAlltoAll(std::shared_ptr<CommunicatorWrapper> comm,
                                   void* input_buff,
                                   void* output_buff,
                                   const size_t buff_size,
                                   cudaStream_t stream,
                                   int sm_num,
                                   int block_size,
                                   const int nranks,
                                   const int rank,
                                   std::vector<int>& ranks_send,
                                   std::vector<int>& ranks_recv) {
  auto& connections = init_connections(comm->comm, rank, nranks);
  NetAlltoAllUneven wrapper;
  init_NetAlltoAllUneven(wrapper,
                         comm,
                         connections,
                         rank,
                         nranks,
                         input_buff,
                         output_buff,
                         buff_size,
                         stream,
                         sm_num,
                         block_size,
                         ranks_send,
                         ranks_recv);
  wrapper(wrapper.stream, wrapper.nblocks, wrapper.nthreads, wrapper.sync_mode);
}

void Custom_MScclpp_AlltoAll(std::shared_ptr<CommunicatorWrapper> comm,
                             void* input_buff,
                             void* output_buff,
                             const size_t buff_size,
                             cudaStream_t stream,
                             int sm_num,
                             int block_size,
                             int nranks,
                             int rank) {
  auto& connections = init_connections(comm->comm, rank, nranks);
  NetAlltoAll wrapper;
  init_NetAlltoAll(
      wrapper, comm, connections, rank, nranks, input_buff, output_buff, buff_size, stream, sm_num, block_size);
  wrapper(wrapper.stream, wrapper.nblocks, wrapper.nthreads, wrapper.sync_mode);
  // CUDA_CHECK(cudaDeviceSynchronize());
}

void Custom_MScclpp_AlltoAllv(std::shared_ptr<CommunicatorWrapper> comm,
                              void* input_buff,
                              void* output_buff,
                              const size_t input_size,
                              const size_t output_size,
                              cudaStream_t stream,
                              int sm_num,
                              int block_size,
                              const int nranks,
                              const int rank,
                              std::vector<int>& input_lengths,
                              std::vector<int>& input_offsets,
                              std::vector<std::vector<int>>& output_lengths_all,
                              std::vector<std::vector<int>>& output_offsets_all) {
  auto& connections = init_connections(comm->comm, rank, nranks);
  NetAlltoAllv wrapper;
  init_NetAlltoAllv(wrapper,
                    comm,
                    connections,
                    rank,
                    nranks,
                    input_buff,
                    output_buff,
                    input_size,
                    output_size,
                    stream,
                    sm_num,
                    block_size,
                    input_lengths,
                    input_offsets,
                    output_lengths_all,
                    output_offsets_all);
  wrapper(wrapper.stream, wrapper.nblocks, wrapper.nthreads, wrapper.sync_mode);
  // CUDA_CHECK(cudaDeviceSynchronize());
}

// Add a cleanup function to clear connections when needed
void cleanup_connections(std::shared_ptr<mscclpp::Communicator> comm) {
  connection_cache.erase(comm);
}

void CommunicatorWrapper::init_NetAlltoAllv_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                                    const int rank,
                                                    const int nranks,
                                                    cudaStream_t stream) {
  size_t head_num = 24;
  size_t head_dim = 128;
  std::vector<size_t> batch_size_list = {1, 2, 4, 8, 16, 32};
  std::vector<size_t> seq_len_list = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
  // std::vector<size_t> batch_size_list = {1};
  // std::vector<size_t> seq_len_list = {1024};

  size_t max_batch_size = *std::max_element(batch_size_list.begin(), batch_size_list.end());
  size_t max_seq_len = *std::max_element(seq_len_list.begin(), seq_len_list.end());
  size_t max_size = max_batch_size * max_seq_len * head_num * head_dim;

  // create a supper large buffer for communication
  void *input_buffer, *output_buffer;
  CUDA_CHECK(cudaMalloc(&input_buffer, max_size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&output_buffer, max_size * sizeof(cutlass::half_t)));

  auto& connections = init_connections(comm, rank, nranks);

  // create a new wrapper
  this->wrapper_sptr = std::make_shared<NetAlltoAllv>();
  bool sync = true;
  int sm_num = nranks - 1;
  int block_size = 512;
  this->wrapper_sptr->setStream(stream);
  this->wrapper_sptr->configRun(sm_num, block_size, sync);
  this->wrapper_sptr->init(
      comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR});
}

void CommunicatorWrapper::init_NetAllReduce_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                                    // void* input_buff,
                                                    // void* output_buff,
                                                    // const size_t max_size,
                                                    const int rank,
                                                    const int nranks,
                                                    cudaStream_t stream) {
  size_t head_num = 32;
  size_t head_dim = 128;
  std::vector<size_t> batch_size_list = {1, 2, 4, 8, 16};
  std::vector<size_t> seq_len_list = {1024, 2048, 4096, 8192};
  // std::vector<size_t> batch_size_list = {1};
  // std::vector<size_t> seq_len_list = {1024};

  size_t max_batch_size = *std::max_element(batch_size_list.begin(), batch_size_list.end());
  size_t max_seq_len = *std::max_element(seq_len_list.begin(), seq_len_list.end());
  size_t max_size = max_batch_size * max_seq_len * head_num * head_dim;

  // create a supper large buffer for communication
  void *input_buffer, *output_buffer;
  CUDA_CHECK(cudaMalloc(&input_buffer, max_size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&output_buffer, max_size * sizeof(cutlass::half_t)));

  auto& connections = init_connections(comm, rank, nranks);

  // create a new wrapper
  this->ar_wrapper_sptr = std::make_shared<NetAllReduce>();
  bool sync = true;
  int sm_num = nranks - 1;
  int block_size = 512;
  this->ar_wrapper_sptr->setStream(stream);
  this->ar_wrapper_sptr->configRun(sm_num, block_size, sync);
  this->ar_wrapper_sptr->init(
      comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR});
}

void CommunicatorWrapper::init_NetAllReduce_wrapper_bf16(std::shared_ptr<mscclpp::Communicator> comm,
                                                         const int rank,
                                                         const int nranks,
                                                         cudaStream_t stream) {
  size_t head_num = 32;
  size_t head_dim = 128;
  std::vector<size_t> batch_size_list = {1, 2, 4, 8, 16};
  std::vector<size_t> seq_len_list = {1024, 2048, 4096, 8192};

  size_t max_batch_size = *std::max_element(batch_size_list.begin(), batch_size_list.end());
  size_t max_seq_len = *std::max_element(seq_len_list.begin(), seq_len_list.end());
  size_t max_size = max_batch_size * max_seq_len * head_num * head_dim;

  void *input_buffer, *output_buffer;
  CUDA_CHECK(cudaMalloc(&input_buffer, max_size * sizeof(cutlass::bfloat16_t)));
  CUDA_CHECK(cudaMalloc(&output_buffer, max_size * sizeof(cutlass::bfloat16_t)));

  auto& connections = init_connections(comm, rank, nranks);

  this->ar_wrapper_sptr = std::make_shared<NetAllReduce>();
  this->ar_wrapper_sptr->bf16_mode = true;
  bool sync = true;
  int sm_num = nranks - 1;
  int block_size = 512;
  this->ar_wrapper_sptr->setStream(stream);
  this->ar_wrapper_sptr->configRun(sm_num, block_size, sync);
  this->ar_wrapper_sptr->init(
      comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buffer, max_size, (size_t)1, PllmLayout::ROW_MAJOR});
}

void CommunicatorWrapper::init_NetAllReduce_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                                    void* input_buff,
                                                    void* output_buff,
                                                    const size_t max_size,
                                                    const int rank,
                                                    const int nranks,
                                                    cudaStream_t stream) {

  auto& connections = init_connections(comm, rank, nranks);

  // create a new wrapper
  this->ar_wrapper_sptr = std::make_shared<NetAllReduce>();
  bool sync = true;
  int sm_num = nranks - 1;
  int block_size = 512;
  this->ar_wrapper_sptr->setStream(stream);
  this->ar_wrapper_sptr->configRun(sm_num, block_size, sync);
  this->ar_wrapper_sptr->init(
  comm,
  connections,
  rank,
  nranks,
  pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, max_size, (size_t)1, PllmLayout::ROW_MAJOR},
  pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, max_size, (size_t)1, PllmLayout::ROW_MAJOR});
}

void CommunicatorWrapper::init_NetAllReduce_wrapper_bf16(std::shared_ptr<mscclpp::Communicator> comm,
                                                         void* input_buff,
                                                         void* output_buff,
                                                         const size_t max_size,
                                                         const int rank,
                                                         const int nranks,
                                                         cudaStream_t stream) {
  auto& connections = init_connections(comm, rank, nranks);

  this->ar_wrapper_sptr = std::make_shared<NetAllReduce>();
  this->ar_wrapper_sptr->bf16_mode = true;
  bool sync = true;
  int sm_num = nranks - 1;
  int block_size = 512;
  this->ar_wrapper_sptr->setStream(stream);
  this->ar_wrapper_sptr->configRun(sm_num, block_size, sync);
  this->ar_wrapper_sptr->init(
      comm,
      connections,
      rank,
      nranks,
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, max_size, (size_t)1, PllmLayout::ROW_MAJOR},
      pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, max_size, (size_t)1, PllmLayout::ROW_MAJOR});
}

// just a wrapper for python binding
void init_NetAlltoAllv_wrapper(std::shared_ptr<CommunicatorWrapper> comm,
                               const int rank,
                               const int nranks,
                               cudaStream_t stream) {
  comm->init_NetAlltoAllv_wrapper(comm->comm, rank, nranks, stream);
}

void init_NetAllReduce_wrapper(std::shared_ptr<CommunicatorWrapper> comm,
                                // void* input_buff,
                                // void* output_buff,
                                // const size_t tensor_size,
                                const int rank,
                                const int nranks,
                                cudaStream_t stream) {
  // comm->init_NetAllReduce_wrapper(comm->comm, input_buff, output_buff, tensor_size, rank, nranks, stream);
  comm->init_NetAllReduce_wrapper(comm->comm, rank, nranks, stream);
}

void init_NetAllReduce_wrapper_bf16(std::shared_ptr<CommunicatorWrapper> comm,
                                    const int rank,
                                    const int nranks,
                                    cudaStream_t stream) {
  comm->init_NetAllReduce_wrapper_bf16(comm->comm, rank, nranks, stream);
}

void init_NetAllReduce_wrapper(std::shared_ptr<CommunicatorWrapper> comm,
                             void* input_buff,
                             void* output_buff,
                             const size_t tensor_size,
                             const int rank,
                             const int nranks,
                             cudaStream_t stream) {
    comm->init_NetAllReduce_wrapper(comm->comm, input_buff, output_buff, tensor_size, rank, nranks, stream);
}

void init_NetAllReduce_wrapper_cached(std::shared_ptr<CommunicatorWrapper> comm,
                                      const int rank,
                                      const int nranks,
                                      cudaStream_t stream) {
    comm->init_NetAllReduce_wrapper(comm->comm, rank, nranks, stream);
}

void init_NetAllReduce_wrapper_bf16(std::shared_ptr<CommunicatorWrapper> comm,
                                    void* input_buff,
                                    void* output_buff,
                                    const size_t tensor_size,
                                    const int rank,
                                    const int nranks,
                                    cudaStream_t stream) {
  comm->init_NetAllReduce_wrapper_bf16(comm->comm, input_buff, output_buff, tensor_size, rank, nranks, stream);
}

void init_NetAllReduce_wrapper_cached_bf16(std::shared_ptr<CommunicatorWrapper> comm,
                                           const int rank,
                                           const int nranks,
                                           cudaStream_t stream) {
  comm->init_NetAllReduce_wrapper_bf16(comm->comm, rank, nranks, stream);
}
