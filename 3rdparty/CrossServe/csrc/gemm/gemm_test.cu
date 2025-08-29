#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "cutlassGemmWrapper.cuh"
#include "tensorLogger.cuh"

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

void test_CutlassGemm(int M, int N, int K) {
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
  CutlassGEMMWrapper<256,
                     128,
                     32,
                     64,
                     64,
                     32,
                     /*split_k*/ 1,
                     /*stages*/ 3,
                     cutlass::layout::RowMajor,
                     cutlass::layout::RowMajor,
                     cutlass::layout::RowMajor>
      wrapper;

  wrapper.setStream(0);
  wrapper.set_shape(M, N, K);
  wrapper.setA(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_act, M, K, PllmLayout::ROW_MAJOR});
  wrapper.setB(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_weight, K, N, PllmLayout::ROW_MAJOR});
  wrapper.setC(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_bias, M, N, PllmLayout::ROW_MAJOR});
  wrapper.setD(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_output, M, N, PllmLayout::ROW_MAJOR});
  wrapper.init(beta);
  wrapper.work();

  bool check = wrapper.checkResult();
  if (check) {
    std::cout << "Check result passed" << std::endl;
  } else {
    std::cout << "Check result failed" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  // // Initialize the MPI environment
  // MPI_Init(&argc, &argv);
  // // Get the number of processes
  // int nranks;
  // MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // // Get the rank of the process
  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  CUDA_CHECK(cudaSetDevice(0));

  //  // Print off a hello world message
  // std::cout << "Hello world from rank " << rank << " out of " << nranks << " ranks" << std::endl;

  test_CutlassGemm(4 * 4096, 3072, 3072);
}
