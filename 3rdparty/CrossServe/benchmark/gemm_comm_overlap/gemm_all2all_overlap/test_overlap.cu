// TODO rewrite them in python after pybind is ready
#include "init.cuh"

/*
mpirun --allow-run-as-root -np 2 build/test_overlap 32 128
nsys profile -f true -o overlap mpirun --allow-run-as-root -np 4 build/test_overlap 3 128
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

  cudaStream_t stream1, stream2;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  GemmWrapper gemm_wrapper;
  init_CutlassGemm(gemm_wrapper, seq_len * batch_size / nranks, head_dim * head_num, head_dim * head_num, stream1);

  NetAlltoAll net_wrapper;
  init_NetAlltoAll(net_wrapper,
                   comm,
                   connections,
                   rank,
                   nranks,
                   batch_size * seq_len * head_num * head_dim * sizeof(cutlass::half_t) / nranks,
                   stream2,
                   SM_num,
                   block_size);

  test_energy_time(rank, net_wrapper, gemm_wrapper);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}
