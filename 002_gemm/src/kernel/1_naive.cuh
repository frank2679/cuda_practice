#pragma once
#include <cuda_runtime.h>

#include "kernel_common.h"

__global__ void kernel_v1_naive(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  int cCol = threadIdx.x + blockIdx.x * blockDim.x;
  int cRow = threadIdx.y + blockIdx.y * blockDim.y;

  if (cCol < N && cRow < M) {
    float tmp = 0.f;
    for (int idx_k = 0; idx_k < K; ++idx_k) {
      tmp += MATRIX_IDX(A, cRow, idx_k, M) * MATRIX_IDX(B, idx_k, cCol, K);
    }

    MATRIX_IDX(C, cRow, cCol, M) =
        alpha * tmp + beta * MATRIX_IDX(C, cRow, cCol, M);
  }
}