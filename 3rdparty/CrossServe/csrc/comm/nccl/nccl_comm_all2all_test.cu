#include <assert.h>
#include <mpi.h>

#include "all2all_kernel.h"
#include "nccl.h"

void TestCustomNcclAllToAllv_all2part(ncclComm_t comm, int rank, int nRanks) {
  // Initialize CUDA stream (only one needed per process)
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Allocate device buffers (only for this rank)
  const size_t count = 4;  // Elements per rank
  const size_t totalSize = count * nRanks * sizeof(int);

  int *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc(&sendbuff, totalSize));
  if (rank < nRanks / 2) {
    CUDACHECK(cudaMalloc(&recvbuff, totalSize * 2));
  } else {
    CUDACHECK(cudaMalloc(&recvbuff, 1));
  }

  // Initialize send buffer with test data
  int* hostSendBuff = (int*)malloc(totalSize);
  for (int j = 0; j < count * nRanks; j++) {
    hostSendBuff[j] = rank * 100 + j;  // Unique pattern for this rank
  }
  CUDACHECK(cudaMemcpy(sendbuff, hostSendBuff, totalSize, cudaMemcpyHostToDevice));
  free(hostSendBuff);

  int *device_sendcounts, *device_recvcounts, *device_sendsizes, *device_recvsizes;

  CUDACHECK(cudaMalloc(&device_sendcounts, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_recvcounts, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_sendsizes, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_recvsizes, nRanks * sizeof(int)));

  int* host_sendcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_recvcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_sendsizes = (int*)malloc(nRanks * sizeof(int));
  int* host_recvsizes = (int*)malloc(nRanks * sizeof(int));

  for (int j = 0; j < nRanks; j++) {
    if (j < nRanks / 2) {
      host_sendcounts[j] = count * 2;
    } else {
      host_sendcounts[j] = 0;
    }

    if (rank < nRanks / 2) {
      host_recvcounts[j] = count * 2;
    } else {
      host_recvcounts[j] = 0;
    }

    if (j == 0) {
      host_sendsizes[j] = 0;
    } else {
      host_sendsizes[j] = count * 2 + host_sendsizes[j - 1];
    }

    if (j == 0) {
      host_recvsizes[j] = 0;
    } else {
      host_recvsizes[j] = count * 2 + host_recvsizes[j - 1];
    }
  }

  ncclResult_t result = CustomNcclAllToAllv(
      sendbuff, host_sendcounts, host_sendsizes, recvbuff, host_recvcounts, host_recvsizes, ncclInt, comm, stream);

  assert(result == ncclSuccess);

  // Cleanup
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  if (rank == 0) {
    printf("AlltoAllv all2part test passed successfully!\n");
  }
}

void TestCustomNcclAllToAllv_part2all(ncclComm_t comm, int rank, int nRanks) {
  // Initialize CUDA stream (only one needed per process)
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Allocate device buffers (only for this rank)
  const size_t count = 4;  // Elements per rank
  const size_t totalSize = count * nRanks * sizeof(int);

  int *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc(&recvbuff, totalSize));
  if (rank < nRanks / 2) {
    CUDACHECK(cudaMalloc(&sendbuff, totalSize * 2));
  } else {
    CUDACHECK(cudaMalloc(&sendbuff, 1));
  }

  // Initialize send buffer with test data
  int* hostSendBuff;
  if (rank < nRanks / 2) {
    hostSendBuff = (int*)malloc(totalSize * 2);
    for (int j = 0; j < 2 * count * nRanks; j++) {
      hostSendBuff[j] = rank * 100 + j;  // Unique pattern for this rank
    }

    CUDACHECK(cudaMemcpy(sendbuff, hostSendBuff, totalSize * 2, cudaMemcpyHostToDevice));
  } else {
    hostSendBuff = (int*)malloc(1);
    CUDACHECK(cudaMemcpy(sendbuff, hostSendBuff, 1, cudaMemcpyHostToDevice));
  }
  free(hostSendBuff);

  int* host_sendcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_recvcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_sendsizes = (int*)malloc(nRanks * sizeof(int));
  int* host_recvsizes = (int*)malloc(nRanks * sizeof(int));

  for (int j = 0; j < nRanks; j++) {
    if (rank < nRanks / 2) {
      host_sendcounts[j] = count * 2;
    } else {
      host_sendcounts[j] = 0;
    }

    if (j < nRanks / 2) {
      host_recvcounts[j] = count * 2;
    } else {
      host_recvcounts[j] = 0;
    }

    if (rank < nRanks / 2) {
      if (j == 0) {
        host_sendsizes[j] = 0;
      } else {
        host_sendsizes[j] = count * 2 + host_sendsizes[j - 1];
      }
    }

    if (j == 0) {
      host_recvsizes[j] = 0;
    } else {
      host_recvsizes[j] = count * 2 + host_recvsizes[j - 1];
    }
  }

  ncclResult_t result = CustomNcclAllToAllv(
      sendbuff, host_sendcounts, host_sendsizes, recvbuff, host_recvcounts, host_recvsizes, ncclInt, comm, stream);

  assert(result == ncclSuccess);

  // Cleanup
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  if (rank == 0) {
    printf("AlltoAllv part2all test passed successfully!\n");
  }
}

void TestCustomNcclAllToAll(ncclComm_t comm, int rank, int nRanks) {
  // Initialize CUDA stream (only one needed per process)
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Allocate device buffers (only for this rank)
  const size_t count = 4;  // Elements per rank
  const size_t totalSize = count * nRanks * sizeof(int);

  int *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc(&sendbuff, totalSize));
  CUDACHECK(cudaMalloc(&recvbuff, totalSize));

  // Initialize send buffer with test data
  int* hostSendBuff = (int*)malloc(totalSize);
  for (int j = 0; j < count * nRanks; j++) {
    hostSendBuff[j] = rank * 100 + j;  // Unique pattern for this rank
  }
  CUDACHECK(cudaMemcpy(sendbuff, hostSendBuff, totalSize, cudaMemcpyHostToDevice));
  free(hostSendBuff);

  int *device_sendcounts, *device_recvcounts, *device_sendsizes, *device_recvsizes;

  CUDACHECK(cudaMalloc(&device_sendcounts, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_recvcounts, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_sendsizes, nRanks * sizeof(int)));
  CUDACHECK(cudaMalloc(&device_recvsizes, nRanks * sizeof(int)));

  int* host_sendcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_recvcounts = (int*)malloc(nRanks * sizeof(int));
  int* host_sendsizes = (int*)malloc(nRanks * sizeof(int));
  int* host_recvsizes = (int*)malloc(nRanks * sizeof(int));

  for (int j = 0; j < nRanks; j++) {
    host_sendcounts[j] = count;
    host_recvcounts[j] = count;
    host_sendsizes[j] = 1;
    host_recvsizes[j] = 1;
  }

  CUDACHECK(cudaMemcpy(device_sendcounts, host_sendcounts, nRanks * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(device_recvcounts, host_recvcounts, nRanks * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(device_sendsizes, host_sendsizes, nRanks * sizeof(int), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(device_recvsizes, host_recvsizes, nRanks * sizeof(int), cudaMemcpyHostToDevice));

  // Run AlltoAll
  ncclResult_t result = CustomNcclAlltoAll(sendbuff, recvbuff, count, ncclInt, ncclSum, comm, stream);

  assert(result == ncclSuccess);

  // Verify results
  int* hostRecvBuff = (int*)malloc(totalSize);
  CUDACHECK(cudaMemcpy(hostRecvBuff, recvbuff, totalSize, cudaMemcpyDeviceToHost));

  // Verify the data received by this rank
  for (int j = 0; j < nRanks; j++) {
    for (int k = 0; k < count; k++) {
      int expected = j * 100 + (rank * count + k);
      if (hostRecvBuff[j * count + k] != expected) {
        printf("Rank %d: Validation failed at position %d. Expected %d, got %d\n",
               rank,
               j * count + k,
               expected,
               hostRecvBuff[j * count + k]);
        assert(0);
      }
    }
  }
  free(hostRecvBuff);

  // Cleanup
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  CUDACHECK(cudaStreamDestroy(stream));

  if (rank == 0) {
    printf("AlltoAll test passed successfully!\n");
  }
}

/*
mpirun --allow-run-as-root -np 4 ./build/test_nccl_all2all
*/

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get MPI rank and size
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Set GPU device based on rank
  CUDACHECK(cudaSetDevice(rank));

  // Initialize NCCL
  ncclUniqueId id;
  if (rank == 0)
    ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Create NCCL communicator
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  TestCustomNcclAllToAll(comm, rank, size);

  MPI_Barrier(MPI_COMM_WORLD);

  TestCustomNcclAllToAllv_all2part(comm, rank, size);

  MPI_Barrier(MPI_COMM_WORLD);

  TestCustomNcclAllToAllv_part2all(comm, rank, size);

  // Cleanup
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}
