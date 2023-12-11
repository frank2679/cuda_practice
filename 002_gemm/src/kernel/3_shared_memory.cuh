#pragma once
#include <cuda_runtime.h>

#include "kernel_common.h"

template <int BLOCK_SIZE>
__global__ void kernel_v3_shared_memory(int M, int N, int K, float alpha,
                                        float *A, float *B, float beta,
                                        float *C) {
  int cRow = threadIdx.x + blockIdx.x * blockDim.x;
  int cCol = threadIdx.y + blockIdx.y * blockDim.y;
  int threadRow = threadIdx.x;
  int threadCol = threadIdx.y;
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  // advance to the block to compute
  A += blockRow * BLOCK_SIZE;
  B += blockCol * BLOCK_SIZE * K;
  C += blockCol * BLOCK_SIZE * M + blockRow * BLOCK_SIZE;

  if (cCol < N && cRow < M) {
    float tmp = 0.f;
    for (int idx_blk = 0; idx_blk < K; idx_blk += BLOCK_SIZE) {
      // load to shared memory, each thread load one element of A, B
      // also coalescing
      MATRIX_IDX(As, threadRow, threadCol, BLOCK_SIZE) =
          MATRIX_IDX(A, threadRow, threadCol, M);
      MATRIX_IDX(Bs, threadRow, threadCol, BLOCK_SIZE) =
          MATRIX_IDX(B, threadRow, threadCol, K);

      // block threads in this block until cache is fully populated
      __syncthreads();

      A += BLOCK_SIZE * M;
      B += BLOCK_SIZE;

      // compute
      for (int idx_k = 0; idx_k < BLOCK_SIZE; ++idx_k) {
        tmp += MATRIX_IDX(As, threadRow, idx_k, BLOCK_SIZE) *
               MATRIX_IDX(Bs, idx_k, threadCol, BLOCK_SIZE);
      }

      // need to sync again at the end, to avoid faster threads
      // fetching the next block into the cache before slower threads are done
      __syncthreads();
    }

    MATRIX_IDX(C, threadRow, threadCol, M) =
        alpha * tmp + beta * MATRIX_IDX(C, threadRow, threadCol, M);
  }
}
