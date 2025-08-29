#include <stdio.h>

#include "all2all_kernel.h"

template <typename T>
inline bool _nccl_should_send_recv(T value) {
  return value != 0;
}

#ifdef __cplusplus
extern "C" {
#endif

ncclResult_t CustomNcclAlltoAll(void* sendbuff,
                                void* recvbuff,
                                size_t count,
                                ncclDataType_t type,
                                ncclRedOp_t op,
                                ncclComm_t comm,
                                cudaStream_t stream) {
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoall. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return ncclInvalidArgument;
#else
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < nRanks; r++) {
    NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
#endif
}

ncclResult_t CustomNcclAllToAllv(void* sendbuff,
                                 const int* sendcounts,
                                 const int* sendsizes,
                                 void* recvbuff,
                                 const int* recvcounts,
                                 const int* recvsizes,
                                 ncclDataType_t datatype,
                                 ncclComm_t comm,
                                 cudaStream_t stream) {
  int numranks = 0;
  NCCLCHECK(ncclCommCount(comm, &numranks));
  int rank = 0;
  NCCLCHECK(ncclCommUserRank(comm, &rank));
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < numranks; r++) {
    if (_nccl_should_send_recv(sendcounts[r])) {
      // printf("rank %d send %d bytes at offset %d to %d\n", rank, sendcounts[r] * wordSize(datatype), sendsizes[r] *
      // wordSize(datatype), r);
      NCCLCHECK(
          ncclSend(((char*)sendbuff) + sendsizes[r] * wordSize(datatype), sendcounts[r], datatype, r, comm, stream));
    }
    if (_nccl_should_send_recv(recvcounts[r])) {
      // printf("rank %d recv %d bytes at offset %d from %d\n", rank, recvcounts[r] * wordSize(datatype), recvsizes[r] *
      // wordSize(datatype), r);
      NCCLCHECK(
          ncclRecv(((char*)recvbuff) + recvsizes[r] * wordSize(datatype), recvcounts[r], datatype, r, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());
  // printf("rank %d finished alltoall\n", rank);
  return ncclSuccess;
}

#ifdef __cplusplus
}
#endif
