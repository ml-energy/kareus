#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <helper.h>
#include <nvml.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<__blockM__, __blockN__, __blockK__>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<__warpM__, __warpN__, __warpK__>;
using ShapeMMAOp = cutlass::gemm::GemmShape<__instM__, __instN__, __instK__>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                                128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                                ElementAccumulator,
                                                                ElementComputeEpilogue,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

int constexpr NumStages = __kstages__;

int constexpr AlignmentA = 8;
int constexpr AlignmentB = 8;

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator,
                                                  MMAOp,
                                                  SmArch,
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages,
                                                  AlignmentA,
                                                  AlignmentB,
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  false,                      /*GatherA*/
                                                  false,                      /*GatherB*/
                                                  false,                      /*ScatterD*/
                                                  cutlass::layout::NoPermute, /*PermuteDLayout*/
                                                  cutlass::layout::NoPermute, /*PermuteALayout*/
                                                  cutlass::layout::NoPermute  /*PermuteBLayout*/
                                                  >;

struct CudaBuffer {
  void* _data;
  int _size;

  CudaBuffer(int size_in_bytes) : _size(size_in_bytes) {
    cudaMalloc(&_data, _size);
  }

  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(_data);
  }

  void copy_to(void* dst) {
    cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
  }

  void copy_from(void* src) {
    cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
  }

  ~CudaBuffer() {
    cudaFree(_data);
  }
};

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

int main() {
  int length_m = __m__;
  int length_n = __n__;
  int length_k = __k__;
  int batch = __batch__;

  CudaBuffer d_act(batch * length_m * length_k * sizeof(__half));
  CudaBuffer d_weight(length_k * length_n * sizeof(__half));
  CudaBuffer d_out(batch * length_m * length_n * sizeof(__half));
  CudaBuffer d_bias(length_n * sizeof(__half));
  std::vector<__half> h_act(batch * length_m * length_k);
  std::vector<__half> h_weight(length_k * length_n);
  std::vector<__half> h_bias(length_n);

  random_fill(h_act, -1.f, 1.f);
  random_fill(h_weight, -1.f, 1.f);
  random_fill(h_bias, -1.f, 1.f);

  d_act.copy_from(h_act.data());
  d_weight.copy_from(h_weight.data());
  d_bias.copy_from(h_bias.data());

  float alpha = 1.0;
  float beta = 0.0;
  bool b_broadcast = true;
  bool c_broadcast = true;

  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  typename EpilogueOp::Params epilogue(alpha, beta);

  cutlass::MatrixCoord extent_A{problem_size.m(), problem_size.k()};
  cutlass::MatrixCoord extent_B{problem_size.k(), problem_size.n()};
  cutlass::MatrixCoord extent_C{problem_size.m(), problem_size.n()};

  LayoutInputA layout_A(LayoutInputA::packed(extent_A));
  LayoutInputB layout_B(LayoutInputB::packed(extent_B));
  LayoutOutput layout_C(LayoutOutput::packed(extent_C));

  typename Gemm::Arguments arguments{
      batch > 1 ? cutlass::gemm::GemmUniversalMode::kBatched : cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      batch,
      epilogue,
      (void*)d_act.data(),
      (void*)d_weight.data(),
      (void*)d_bias.data(),
      (void*)d_out.data(),
      layout_A.capacity(extent_A),
      b_broadcast ? 0 : layout_B.capacity(extent_B),
      c_broadcast ? 0 : layout_C.capacity(extent_C),
      layout_C.capacity(extent_C),
      layout_A.stride(0),
      layout_B.stride(0),
      c_broadcast ? 0 : layout_C.stride(0),
      layout_C.stride(0),
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace_data;
  cudaMalloc(&workspace_data, workspace_size);

  Gemm gemm_op;
  gemm_op.initialize(arguments, workspace_data);

  for (int i = 0; i < 10; i++) {
    gemm_op();
  }

  cudaEvent_t warmup_start, warmup_stop;
  cudaEventCreate(&warmup_start);
  cudaEventCreate(&warmup_stop);

  cudaEventRecord(warmup_start, 0);

  for (int i = 0; i < 10; i++) {
    gemm_op();
  }

  cudaEventRecord(warmup_stop, 0);
  cudaEventSynchronize(warmup_stop);
  float warmup_elapsed;
  cudaEventElapsedTime(&warmup_elapsed, warmup_start, warmup_stop);

  int iterations = (int)(10000 / (warmup_elapsed / 10));

  cudaEvent_t start, start_, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&start_);
  cudaEventCreate(&stop);

  nvmlReturn_t result;
  result = nvmlInit();
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
    return 1;
  }
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(2, &device);
  unsigned long long energy_consumed_start, energy_consumed_end;

  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_consumed_start);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to get energy consumption for device: " << nvmlErrorString(result) << std::endl;
  }

  cudaEventRecord(start_, 0);

  for (int i = 0; i < iterations; i++) {
    gemm_op();
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  result = nvmlDeviceGetTotalEnergyConsumption(device, &energy_consumed_end);
  if (result != NVML_SUCCESS) {
    std::cerr << "Failed to get energy consumption for device: " << nvmlErrorString(result) << std::endl;
  }

  float elapsed;
  cudaEventElapsedTime(&elapsed, start_, stop);

  printf("time: %f ms\n", elapsed / iterations);
  printf("energy: %f mJ\n", (double)(energy_consumed_end - energy_consumed_start) / (double)iterations);
}
