#include <mpi.h>
#include <nvml.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "netWrapper.cuh"
#include "tensorLogger.cuh"

void test_NetAlltoAll(std::shared_ptr<mscclpp::Communicator> comm,
                      std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                      const int rank,
                      const int nranks,
                      const size_t buff_size,
                      const bool inplace,
                      const bool columnwise,
                      int sm_num,
                      int block_size) {
  assert(inplace == false);
  assert(columnwise == false);

  // Intialize host and device buffers
  std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
  for (size_t i = 0; i < host_buff.size(); ++i)
    host_buff[i] = cutlass::half_t(int((i * rank) % 101));
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, buff_size));

  // Initialize NetWrapper
  NetAlltoAll wrapper;
  int dim1, input_dim2, output_dim2;
  dim1 = buff_size / sizeof(cutlass::half_t);
  input_dim2 = 1;
  output_dim2 = 1;

  wrapper.init(comm,
               connections,
               rank,
               nranks,
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, dim1, output_dim2, PllmLayout::ROW_MAJOR});

  MPI_Barrier(MPI_COMM_WORLD);
  wrapper.setColumnwise(columnwise);

  cudaEvent_t start, start_, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&start_);
  cudaEventCreate(&stop);

  nvmlReturn_t result;
  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
  }
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(rank, &device);
  unsigned long long energy_consumed_start, energy_consumed_end;

  int warmup_iters = 100;
  int iterations = 1000;

  for (int i = 0; i < warmup_iters; i++) {
    wrapper(0, sm_num, block_size, true);
    // CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_consumed_start);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to get energy consumption for device: " << nvmlErrorString(result) << std::endl;
  }

  cudaEventRecord(start_, 0);

  for (int i = 0; i < iterations; i++) {
    wrapper(0, sm_num, block_size, true);
    // CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_consumed_end);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to get energy consumption for device: " << nvmlErrorString(result) << std::endl;
  }

  float elapsed;
  cudaEventElapsedTime(&elapsed, start_, stop);

  std::cout << "Rank " << rank << ", time: " << elapsed / iterations
            << " ms, energy: " << (double)(energy_consumed_end - energy_consumed_start) / (double)iterations << " mJ"
            << std::endl;

  CUDA_CHECK(cudaFree(input_buff));
  if (input_buff != output_buff)
    CUDA_CHECK(cudaFree(output_buff));
}

/*
mpirun --allow-run-as-root -np 2 csrc/comm/build/test_comm
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

  // Ensure there are enough arguments
  if (argc < 3) {
    if (rank == 0) {  // Only rank 0 prints error messages
      std::cerr << "Usage: " << argv[0] << " <SM number> <block size>\n";
    }
    MPI_Finalize();
    return 1;
  }

  int SM_num = std::atoi(argv[1]);
  int block_size = std::atoi(argv[2]);

  // Print off a hello world message
  // std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;

  // Initialize Communicator
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId uniqueId;
  if (rank == 0)
    uniqueId = bootstrap->createUniqueId();
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(uniqueId);
  auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);

  // Initialize Connections
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

  MPI_Barrier(MPI_COMM_WORLD);

  // Tests
  constexpr size_t buff_size = 4 * 16384 * 24 * 128 / 4 * 2;
  test_NetAlltoAll(comm, connections, rank, nranks, buff_size, false, false, SM_num, block_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
