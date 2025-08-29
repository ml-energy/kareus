#include <mpi.h>
#include <nvml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "cutlassGemmWrapper.cuh"
#include "netWrapper.cuh"
#include "tensorLogger.cuh"

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

void init_NetAlltoAll(NetAlltoAll& wrapper,
                      std::shared_ptr<mscclpp::Communicator> comm,
                      std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                      const int rank,
                      const int nranks,
                      const size_t buff_size,
                      cudaStream_t stream,
                      int sm_num,
                      int block_size) {
  std::vector<cutlass::half_t> host_buff(buff_size / sizeof(cutlass::half_t));
  random_fill(host_buff, -1.f, 1.f);
  void* input_buff;
  CUDA_CHECK(cudaMalloc(&input_buff, buff_size));
  CUDA_CHECK(cudaMemcpy(input_buff, host_buff.data(), buff_size, cudaMemcpyHostToDevice));
  void* output_buff;
  CUDA_CHECK(cudaMalloc(&output_buff, buff_size));

  int dim1, input_dim2, output_dim2;
  dim1 = buff_size / sizeof(cutlass::half_t);
  input_dim2 = 1;
  output_dim2 = 1;

  bool sync = false;
  wrapper.setStream(stream);
  wrapper.configRun(sm_num, block_size, sync);
  wrapper.init(comm,
               connections,
               rank,
               nranks,
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)input_buff, dim1, input_dim2, PllmLayout::ROW_MAJOR},
               pllmTensor<cutlass::half_t>{(cutlass::half_t*)output_buff, dim1, output_dim2, PllmLayout::ROW_MAJOR});

  MPI_Barrier(MPI_COMM_WORLD);
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
  wrapper.setA(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_act, M, K, PllmLayout::ROW_MAJOR});
  wrapper.setB(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_weight, K, N, PllmLayout::ROW_MAJOR});
  wrapper.setC(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_bias, M, N, PllmLayout::ROW_MAJOR});
  wrapper.setD(pllmTensor<cutlass::half_t>{(cutlass::half_t*)d_output, M, N, PllmLayout::ROW_MAJOR});
  wrapper.init(beta);
}

void run(NetAlltoAll& net_wrapper, GemmWrapper& gemm_wrapper) {
  // net_wrapper(0, 32, 1024, true);
  net_wrapper(net_wrapper.stream, net_wrapper.nblocks, net_wrapper.nthreads, net_wrapper.sync_mode);
  gemm_wrapper.work();
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_energy_time(int rank, NetAlltoAll& net_wrapper, GemmWrapper& gemm_wrapper) {
  cudaEvent_t start, start_, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&start_);
  cudaEventCreate(&stop);

  nvmlReturn_t result;
  result = nvmlInit();
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(rank, &device);
  unsigned long long energy_consumed_start, energy_consumed_end;
  int warmup_iters = 300;
  int iterations = 1000;

  for (int i = 0; i < warmup_iters; i++) {
    run(net_wrapper, gemm_wrapper);
  }

  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_consumed_start);
  cudaEventRecord(start_, 0);

  for (int i = 0; i < iterations; i++) {
    run(net_wrapper, gemm_wrapper);
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
}
