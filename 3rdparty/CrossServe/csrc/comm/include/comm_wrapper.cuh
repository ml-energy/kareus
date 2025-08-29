#pragma once

// some helper functions of mscclpp, for python binding
#include <algorithm>
#include <cmath>
#include <mscclpp/core.hpp>
#include <random>

#include "spdlog/spdlog.h"

/*****************************************************/
/*
    CommunicatorWrapper
*/
/*****************************************************/

class NetAlltoAllv;  // cpp is the best language :D
class NetAllReduce;
class NetAllReduceBF16;

class CommunicatorWrapper {
  private:
  // TODO: make all data members private and add public getters, but very low priority
  public:
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  int rank;
  int nranks;
  std::shared_ptr<NetAlltoAllv> wrapper_sptr = nullptr;
  std::shared_ptr<NetAllReduce> ar_wrapper_sptr = nullptr;

  CommunicatorWrapper(std::shared_ptr<mscclpp::Communicator> comm,
                      std::vector<std::shared_ptr<mscclpp::Connection>>& connections,
                      int rank,
                      int nranks)
      : comm(comm), connections(connections), rank(rank), nranks(nranks), wrapper_sptr(nullptr) {
    // init_NetAlltoAllv_wrapper(comm, rank, nranks, stream); // no stream here
  }

  void init_NetAlltoAllv_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                 const int rank,
                                 const int nranks,
                                 cudaStream_t stream);
  
  void init_NetAllReduce_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                // void* input_buffer,
                                // void* output_buffer,
                                // const size_t max_size,
                                const int rank,
                                const int nranks,
                                cudaStream_t stream);
  
  void init_NetAllReduce_wrapper(std::shared_ptr<mscclpp::Communicator> comm,
                                void* input_buffer,
                                void* output_buffer,
                                const size_t max_size,
                                const int rank,
                                const int nranks,
                                cudaStream_t stream);

  // BF16 variants
  void init_NetAllReduce_wrapper_bf16(std::shared_ptr<mscclpp::Communicator> comm,
                                      const int rank,
                                      const int nranks,
                                      cudaStream_t stream);

  void init_NetAllReduce_wrapper_bf16(std::shared_ptr<mscclpp::Communicator> comm,
                                      void* input_buff,
                                      void* output_buff,
                                      const size_t max_size,
                                      const int rank,
                                      const int nranks,
                                      cudaStream_t stream);
};
