#ifndef ALL2ALL_KERNEL_H
#define ALL2ALL_KERNEL_H

#include <nccl.h>

#include "common.h"

#define CUDACHECK(cmd)                                                                        \
  do {                                                                                        \
    cudaError_t err = cmd;                                                                    \
    if (err != cudaSuccess) {                                                                 \
      printf("Failed: CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

#define NCCLCHECK(cmd)                   \
  do {                                   \
    ncclResult_t res = cmd;              \
    if (res != ncclSuccess) {            \
      char hostname[1024];               \
      getHostName(hostname, 1024);       \
      printf(                            \
          "%s: Test NCCL failure %s:%d " \
          "'%s / %s'\n",                 \
          hostname,                      \
          __FILE__,                      \
          __LINE__,                      \
          ncclGetErrorString(res),       \
          ncclGetLastError(NULL));       \
      return ncclInvalidArgument;        \
    }                                    \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t CustomNcclAlltoAll(void* sendbuff,
                                void* recvbuff,
                                size_t count,
                                ncclDataType_t type,
                                ncclRedOp_t op,
                                ncclComm_t comm,
                                cudaStream_t stream);

ncclResult_t CustomNcclAllToAllv(void* sendbuff,
                                 const int* sendcounts,
                                 const int* sendsizes,
                                 void* recvbuff,
                                 const int* recvcounts,
                                 const int* recvsizes,
                                 ncclDataType_t datatype,
                                 ncclComm_t comm,
                                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // ALL2ALL_KERNEL_H
