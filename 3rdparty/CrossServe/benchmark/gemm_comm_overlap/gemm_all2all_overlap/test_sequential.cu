// TODO rewrite them in python after pybind is ready
#include "init.cuh"

/*
mpirun --allow-run-as-root -np 2 build/test_sequential
nsys profile -f true -o sequential mpirun --allow-run-as-root -np 4 build/test_sequential
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
  int batch_size = 4;
  int seq_len = 4096;
  int head_num = 24;
  int head_dim = 128;

  GemmWrapper gemm_wrapper;
  init_CutlassGemm(gemm_wrapper, seq_len * batch_size / nranks, head_dim * head_num, head_dim * head_num, 0);

  int sm_num_comm = 3;
  int block_size_comm = 512;
  NetAlltoAll net_wrapper;
  init_NetAlltoAll(net_wrapper,
                   comm,
                   connections,
                   rank,
                   nranks,
                   batch_size * seq_len * head_num * head_dim * sizeof(cutlass::half_t) / nranks,
                   0,
                   sm_num_comm,
                   block_size_comm);

  test_energy_time(rank, net_wrapper, gemm_wrapper);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
