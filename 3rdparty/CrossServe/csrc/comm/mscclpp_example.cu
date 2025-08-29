// this is just a example to show how to invoke the mscclpp all2all communication, and the api is aligned with python
// binding api
#include <mpi.h>

#include "include/msccl_init.cuh"

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

/*
mpirun --allow-run-as-root -np 4 ./build/test_mscclpp 3 1024
*/

int main(int argc, char* argv[]) {
  // in cpp side, we use mpi to sync the nccl unique id, but in python level we could use dist.broadcast to sync the
  // unique id

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

  int sm_num = std::atoi(argv[1]);
  int block_size = std::atoi(argv[2]);

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
  size_t buff_size = 1024 * 1024 * 1024;
  std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
  random_fill(host_buff, -1.f, 1.f);
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, buff_size));

  // create a stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // do the all2all communication
  Custom_MScclpp_AlltoAll(comm, input_buff, output_buff, buff_size, stream, sm_num, block_size, nranks, rank);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(host_buff.data(), output_buff, buff_size, cudaMemcpyDeviceToHost));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    spdlog::error("CUDA error: {}", cudaGetErrorString(err));
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
