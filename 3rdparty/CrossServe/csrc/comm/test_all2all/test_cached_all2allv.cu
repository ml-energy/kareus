#include <cuda_runtime.h>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include "debug.h"
#include "msccl_init.cuh"
template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

void test_NetAlltoAll(std::shared_ptr<CommunicatorWrapper> comm, cudaStream_t stream) {
  int nranks = comm->nranks;
  int rank = comm->rank;

  constexpr size_t input_size = 4 * 8;
  constexpr size_t output_size = 4 * 8;
  std::vector<int> input_lengths = {8, 8, 8, 8};
  std::vector<int> input_offests = {0, 8, 16, 24};
  std::vector<std::vector<int>> output_lengths_all, output_offsets_all;
  for (int i = 0; i < nranks; i++) {
    output_lengths_all.push_back({8, 8, 8, 8});
    output_offsets_all.push_back({0, 8, 16, 24});
  }

  // constexpr size_t input_size = 4 * 1024;
  // constexpr size_t output_size = 4 * 1024;
  // std::vector<int> input_lengths = {1024, 1024, 1024, 1024};
  // std::vector<int> input_offests = {0, 1024, 2048, 3072};
  // std::vector<std::vector<int>> output_lengths_all, output_offsets_all;
  // for (int i = 0; i < nranks; i++)
  // {
  //     output_lengths_all.push_back({1024, 1024, 1024, 1024});
  //     output_offsets_all.push_back({0, 1024, 2048, 3072});
  // }

  std::vector<cutlass::half_t> host_buff(input_size);
  for (size_t i = 0; i < host_buff.size(); ++i)
    host_buff[i] = cutlass::half_t(int((i * rank) % 101));
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, input_size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), input_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, output_size * sizeof(cutlass::half_t)));

  // spdlog::info("Rank {} host_buff: {}", rank, vector_to_string(host_buff));

  int sm_num = 3;
  int block_size = 512;

  // cached_Custom_MScclpp_AlltoAllv(comm,
  //                                 input_buff,
  //                                 output_buff,
  //                                 input_size,
  //                                 output_size,
  //                                 stream,
  //                                 sm_num,
  //                                 block_size,
  //                                 // nranks,
  //                                 // rank,
  //                                 input_lengths,
  //                                 input_offests,
  //                                 output_lengths_all,
  //                                 output_offsets_all);
  cached_Custom_MScclpp_AlltoAllv_FuseCopy(comm,
                                           input_buff,
                                           output_buff,
                                           input_size,
                                           output_size,
                                           stream,
                                           sm_num,
                                           block_size,
                                           // nranks,
                                           // rank,
                                           input_lengths,
                                           input_offests,
                                           output_lengths_all,
                                           output_offsets_all);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Check alltoall correctness
  const size_t nelem_per_shard = host_buff.size() / nranks;
  CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, output_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

  // spdlog::info("Rank {} after alltoall host_buff: {}", rank, vector_to_string(host_buff));

  for (size_t i = 0; i < host_buff.size(); ++i) {
    const int remoteRank = i / nelem_per_shard;
    int first_num = rank * nelem_per_shard + i % nelem_per_shard;
    int second_num = remoteRank;
    cutlass::half_t expected = cutlass::half_t(int((first_num * second_num) % 101));
    if (abs(host_buff[i] - expected) > 1e-3) {
      spdlog::error("Rank {} received incorrect data from rank {} at index {}", rank, remoteRank, i);
      break;
    }
  }

  CUDA_CHECK(cudaFree(input_buff));
  if (input_buff != output_buff)
    CUDA_CHECK(cudaFree(output_buff));
  spdlog::info("Rank {} NetAlltoAll test finished", rank);
}

void test_NetAlltoAllv(std::shared_ptr<CommunicatorWrapper> comm, cudaStream_t stream) {
  int nranks = comm->nranks;
  int rank = comm->rank;

  constexpr size_t input_size = 4 * 1024;
  constexpr size_t output_size = 8 * 1024;
  std::vector<int> input_lengths(nranks, 0);
  std::vector<int> input_offsets(nranks, 0);
  std::vector<std::vector<int>> output_lengths_all, output_offsets_all;

  /*
  MLP4->U1R2:
  rank0: input_split_size: {1, 0, 0, 0}, output split size{1, 1, 0, 0}}
  rank1: input_split_size: {1, 0, 0, 0}, output split size{0, 0, 1, 1}}
  rank2: input_split_size: {0, 1, 0, 0}, output split size{0, 0, 0, 0}}
  rank3: input_split_size: {0, 1, 0, 0}, output split size{0, 0, 0, 0}}
  */
  if (rank == 0) {
    input_lengths = {4096, 0, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 1) {
    input_lengths = {4096, 0, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 2) {
    input_lengths = {0, 4096, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 3) {
    input_lengths = {0, 4096, 0, 0};
    input_offsets = {0, 0, 0, 0};
  }
  // rank0
  output_lengths_all.push_back({4096, 4096, 0, 0});
  output_offsets_all.push_back({0, 4096, 0, 0});
  // rank1
  output_lengths_all.push_back({0, 0, 4096, 4096});
  output_offsets_all.push_back({0, 0, 0, 4096});
  // rank2
  output_lengths_all.push_back({0, 0, 0, 0});
  output_offsets_all.push_back({0, 0, 0, 0});
  // rank3
  output_lengths_all.push_back({0, 0, 0, 0});
  output_offsets_all.push_back({0, 0, 0, 0});

  std::vector<cutlass::half_t> host_buff(input_size);
  for (size_t i = 0; i < host_buff.size(); ++i)
    host_buff[i] = cutlass::half_t(int((i * rank) % 101));
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, input_size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), input_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, output_size * sizeof(cutlass::half_t)));

  int sm_num = 3;
  int block_size = 256;
  // cached_Custom_MScclpp_AlltoAllv(comm,
  //                                 input_buff,
  //                                 output_buff,
  //                                 input_size,
  //                                 output_size,
  //                                 stream,
  //                                 sm_num,
  //                                 block_size,
  //                                 // nranks,
  //                                 // rank,
  //                                 input_lengths,
  //                                 input_offsets,
  //                                 output_lengths_all,
  //                                 output_offsets_all);
  cached_Custom_MScclpp_AlltoAllv_FuseCopy(comm,
                                           input_buff,
                                           output_buff,
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

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int> ranks_send = {0, 1, 2, 3};
  std::vector<int> ranks_recv = {0, 1};
  if (rank < 2) {
    std::vector<cutlass::half_t> output_host_buff(output_size);
    CUDA_CHECK(cudaMemcpy(
        output_host_buff.data(), output_buff, output_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < output_host_buff.size(); ++i) {
      int current_rank = rank;
      int remote_rank = current_rank * 2 + i / 4096;
      cutlass::half_t expected = cutlass::half_t(int(((i % 4096) * remote_rank) % 101));

      if (abs(output_host_buff[i] - expected) > 1e-3) {
        spdlog::error("Rank {} received incorrect data from rank {} at index {}", rank, remote_rank, i);
        break;
      }
    }
  }

  CUDA_CHECK(cudaFree(input_buff));
  if (input_buff != output_buff)
    CUDA_CHECK(cudaFree(output_buff));
  spdlog::info("Rank {} NetAlltoAllv test finished", rank);
}

/*
mpirun --allow-run-as-root -np 4 csrc/comm/build/test_cached_mscclpp_all2allv
*/

int main(int argc, char* argv[]) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  // Get the number of processes
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CUDA_CHECK(cudaSetDevice(rank));

  mscclpp::UniqueId uniqueId;
  // init mscclpp communicator
  init_bootstrap(rank, nranks);
  if (rank == 0) {
    uniqueId = get_unique_id();
  }

  // we could use dist.broadcast_object_list in python side to broadcast the unique id
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

  // get the communicator
  std::shared_ptr<CommunicatorWrapper> comm = mscclppCommInitRank(nranks, uniqueId, rank);

  // init the data
  // size_t buff_size = (4 * 4096 * 24 * 128) * sizeof(cutlass::half_t);
  // std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
  // random_fill(host_buff, -1.f, 1.f);
  // void* input_buff;
  // CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
  // CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
  // void* output_buff;
  // CUDA_CHECK(cudaMalloc(&output_buff, buff_size));

  // create a stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Tests
  comm->init_NetAlltoAllv_wrapper(comm->comm, rank, nranks, stream);

  test_NetAlltoAll(comm, stream);
  test_NetAlltoAllv(comm, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("CUDA error: {}", cudaGetErrorString(err));
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
