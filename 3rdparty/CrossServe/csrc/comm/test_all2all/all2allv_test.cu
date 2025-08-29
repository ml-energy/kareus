#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "msccl_init.cuh"
#include "netWrapper.cuh"
#include "spdlog/spdlog.h"
#include "tensorLogger.cuh"

void test_NetAlltoAll(std::shared_ptr<CommunicatorWrapper> comm) {
  int nranks = comm->nranks;
  int rank = comm->rank;

  constexpr size_t input_size = 4 * 1024;
  constexpr size_t output_size = 4 * 1024;
  std::vector<int> input_lengths = {1024, 1024, 1024, 1024};
  std::vector<int> input_offests = {0, 1024, 2048, 3072};
  std::vector<std::vector<int>> output_lengths_all, output_offsets_all;
  for (int i = 0; i < nranks; i++) {
    output_lengths_all.push_back({1024, 1024, 1024, 1024});
    output_offsets_all.push_back({0, 1024, 2048, 3072});
  }

  std::vector<cutlass::half_t> host_buff(input_size);
  for (size_t i = 0; i < host_buff.size(); ++i)
    host_buff[i] = cutlass::half_t(int((i * rank) % 101));
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, input_size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), input_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, output_size * sizeof(cutlass::half_t)));

  int sm_num = 3;
  int block_size = 512;
  Custom_MScclpp_AlltoAllv(comm,
                           input_buff,
                           output_buff,
                           input_size,
                           output_size,
                           0,
                           sm_num,
                           block_size,
                           nranks,
                           rank,
                           input_lengths,
                           input_offests,
                           output_lengths_all,
                           output_offsets_all);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Check alltoall correctness
  const size_t nelem_per_shard = host_buff.size() / nranks;
  CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, output_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < host_buff.size(); ++i) {
    const int remoteRank = i / nelem_per_shard;
    int first_num = rank * nelem_per_shard + i % nelem_per_shard;
    int second_num = remoteRank;
    cutlass::half_t expected = cutlass::half_t(int((first_num * second_num) % 101));
    if (abs(host_buff[i] - expected) > 1e-3) {
      std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank << " at index " << i
                << std::endl;
      break;
    }
  }

  CUDA_CHECK(cudaFree(input_buff));
  if (input_buff != output_buff)
    CUDA_CHECK(cudaFree(output_buff));
  spdlog::info("Rank {} NetAlltoAll test finished", rank);
}

int rank_idx(int rank, std::vector<int>& ranks) {
  for (int i = 0; i < ranks.size(); i++) {
    if (rank == ranks[i]) {
      return i;
    }
  }
  return -1;
}

void test_NetAlltoAllUneven(std::shared_ptr<CommunicatorWrapper> comm) {
  int nranks = comm->nranks;
  int rank = comm->rank;

  constexpr size_t input_size = 4 * 1024;
  constexpr size_t output_size = 8 * 1024;
  std::vector<int> input_lengths = {2048, 2048, 0, 0};
  std::vector<int> input_offests = {0, 2048, 0, 0};
  std::vector<std::vector<int>> output_lengths_all, output_offsets_all;

  // rank0
  output_lengths_all.push_back({2048, 2048, 2048, 2048});
  output_offsets_all.push_back({0, 2048, 4096, 6144});
  // rank1
  output_lengths_all.push_back({2048, 2048, 2048, 2048});
  output_offsets_all.push_back({0, 2048, 4096, 6144});
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
  Custom_MScclpp_AlltoAllv(comm,
                           input_buff,
                           output_buff,
                           input_size,
                           output_size,
                           0,
                           sm_num,
                           block_size,
                           nranks,
                           rank,
                           input_lengths,
                           input_offests,
                           output_lengths_all,
                           output_offsets_all);

  CUDA_CHECK(cudaDeviceSynchronize());

  const size_t nelem_per_shard = 2048;
  std::vector<int> ranks_send = {0, 1, 2, 3};
  std::vector<int> ranks_recv = {0, 1};
  if (rank < 2) {
    std::vector<cutlass::half_t> output_host_buff(output_size);
    CUDA_CHECK(cudaMemcpy(
        output_host_buff.data(), output_buff, output_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < output_host_buff.size(); ++i) {
      const int remoteRank = ranks_send[i / nelem_per_shard];
      int first_num = rank_idx(rank, ranks_recv) * nelem_per_shard + i % nelem_per_shard;
      int second_num = remoteRank;
      cutlass::half_t expected = cutlass::half_t(int((first_num * second_num) % 101));
      if (abs(output_host_buff[i] - expected) > 1e-3) {
        std::cerr << "Rank " << rank << " received incorrect data from rank " << remoteRank << " at index " << i
                  << std::endl;
        break;
      }
    }
  }

  CUDA_CHECK(cudaFree(input_buff));
  if (input_buff != output_buff)
    CUDA_CHECK(cudaFree(output_buff));
  spdlog::info("Rank {} NetAlltoAllUneven test finished", rank);
}

void test_NetAlltoAllv(std::shared_ptr<CommunicatorWrapper> comm) {
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
  Custom_MScclpp_AlltoAllv(comm,
                           input_buff,
                           output_buff,
                           input_size,
                           output_size,
                           0,
                           sm_num,
                           block_size,
                           nranks,
                           rank,
                           input_lengths,
                           input_offsets,
                           output_lengths_all,
                           output_offsets_all);

  CUDA_CHECK(cudaDeviceSynchronize());

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
mpirun --allow-run-as-root -np 4 csrc/comm/build/test_mscclpp_all2allv
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

  // Print off a hello world message
  spdlog::info("Hello world from rank {} out of {} ranks", rank, nranks);

  // Initialize Communicator
  init_bootstrap(rank, nranks);
  mscclpp::UniqueId uniqueId;
  if (rank == 0)
    uniqueId = get_unique_id();
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  std::shared_ptr<CommunicatorWrapper> comm = mscclppCommInitRank(nranks, uniqueId, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // Tests
  test_NetAlltoAll(comm);
  test_NetAlltoAllUneven(comm);
  test_NetAlltoAllv(comm);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("CUDA error: {}", cudaGetErrorString(err));
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
