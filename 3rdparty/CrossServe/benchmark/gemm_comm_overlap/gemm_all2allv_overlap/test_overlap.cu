#include <mpi.h>

#include "cutlassGemmWrapper.cuh"
#include "msccl_init.cuh"

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

using GemmWrapper = CutlassGEMMWrapper<256,
                                       128,
                                       32,
                                       64,
                                       64,
                                       32,
                                       /*split_k*/ 1,
                                       /*stages*/ 3,
                                       cutlass::layout::RowMajor,
                                       cutlass::layout::RowMajor,
                                       cutlass::layout::RowMajor>;

void init_CutlassGemm(GemmWrapper& wrapper, int M, int N, int K, cudaStream_t stream) {
  std::vector<cutlass::half_t> h_act(M * K);
  std::vector<cutlass::half_t> h_weight(K * N);
  std::vector<cutlass::half_t> h_bias(M * N);
  random_fill(h_act, -1.f, 1.f);
  random_fill(h_weight, -1.f, 1.f);
  random_fill(h_bias, -1.f, 1.f);
  void *d_act, *d_weight, *d_bias, *d_output;
  CUDA_CHECK(cudaMalloc(&d_act, M * K * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&d_weight, K * N * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&d_bias, M * N * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMemcpy(d_act, h_act.data(), M * K * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), K * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), M * N * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));

  cutlass::half_t beta(1);
  wrapper.setStream(stream);
  wrapper.set_shape(M, N, K);
  // A @ B + C = D
  wrapper.setA(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_act, M, K, PllmLayout::ROW_MAJOR});
  wrapper.setB(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_weight, K, N, PllmLayout::ROW_MAJOR});
  wrapper.setC(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_bias, M, N, PllmLayout::ROW_MAJOR});
  wrapper.setD(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_output, M, N, PllmLayout::ROW_MAJOR});
  wrapper.init(beta);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CUDA_CHECK(cudaSetDevice(rank));

  if (argc < 3) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0] << " <SM number> <block size>\n";
    }
    MPI_Finalize();
    return 1;
  }

  int sm_num = std::atoi(argv[1]);
  int block_size = std::atoi(argv[2]);

  // Tests
  int batch_size = 4;
  int seq_len = 16384;
  int head_num = 24;
  int head_dim = 128;

  cudaStream_t stream1, stream2;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  GemmWrapper gemm_wrapper;
  init_CutlassGemm(gemm_wrapper, seq_len * batch_size / nranks, head_dim * head_num, head_dim * head_num, stream1);

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
  comm->init_NetAlltoAllv_wrapper(comm->comm, rank, nranks, stream2);

  int size = seq_len * batch_size * head_dim * head_num / nranks;
  std::vector<cutlass::half_t> h_input(size);
  std::vector<cutlass::half_t> h_output(size);
  random_fill(h_input, -1.f, 1.f);
  random_fill(h_output, -1.f, 1.f);

  void* d_input;
  void* d_output;
  CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(cutlass::half_t)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_output, h_output.data(), size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));

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
    input_lengths = {size, 0, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 1) {
    input_lengths = {size, 0, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 2) {
    input_lengths = {0, size, 0, 0};
    input_offsets = {0, 0, 0, 0};
  } else if (rank == 3) {
    input_lengths = {0, size, 0, 0};
    input_offsets = {0, 0, 0, 0};
  }
  // rank0
  output_lengths_all.push_back({size, size, 0, 0});
  output_offsets_all.push_back({0, size, 0, 0});
  // rank1
  output_lengths_all.push_back({0, 0, size, size});
  output_offsets_all.push_back({0, 0, 0, size});
  // rank2
  output_lengths_all.push_back({0, 0, 0, 0});
  output_offsets_all.push_back({0, 0, 0, 0});
  // rank3
  output_lengths_all.push_back({0, 0, 0, 0});
  output_offsets_all.push_back({0, 0, 0, 0});

  // use a lambda to do the communication and gemm
  auto do_comm_gemm = [&]() {
    cached_Custom_MScclpp_AlltoAllv(comm,
                                    d_input,
                                    d_output,
                                    size,
                                    size,
                                    stream2,
                                    sm_num,
                                    block_size,
                                    input_lengths,
                                    input_offsets,
                                    output_lengths_all,
                                    output_offsets_all);
    gemm_wrapper.work();
    // CUDA_CHECK(cudaDeviceSynchronize());
  };

  // warmup
  for (int i = 0; i < 100; i++) {
    do_comm_gemm();
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  {
    // test
    for (int i = 0; i < 500; i++) {
      do_comm_gemm();
    }

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  {
    auto& connections = init_connections(comm->comm, rank, nranks);
    NetAlltoAllv wrapper;
    init_NetAlltoAllv(wrapper,
                      comm,
                      connections,
                      rank,
                      nranks,
                      d_input,
                      d_output,
                      size,
                      size,
                      stream2,
                      sm_num,
                      block_size,
                      input_lengths,
                      input_offsets,
                      output_lengths_all,
                      output_offsets_all);

    MPI_Barrier(MPI_COMM_WORLD);

    auto do_pure_comm_gemm = [&]() {
      wrapper(wrapper.stream, wrapper.nblocks, wrapper.nthreads, wrapper.sync_mode);
      gemm_wrapper.work();
      // CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int i = 0; i < 500; i++) {
      do_pure_comm_gemm();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  }
  MPI_Finalize();
  return 0;
}
