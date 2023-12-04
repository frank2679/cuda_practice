#pragma once
#include <cuda_runtime.h>

// col-major
#define MATRIX_IDX(buf, i, j, ld) buf[(i) * (ld) + j]

__global__ void gemm_kernel(int M, int N, int K, float alpha, float *A,
                            float *B, float beta, float *C) {
  int gx = threadIdx.x + blockIdx.x * blockDim.x;
  int gy = threadIdx.y + blockIdx.y * blockDim.y;

  if (gx < N && gy < M) {
    float tmp = 0.f;
    for (int idx_k = 0; idx_k < K; ++idx_k) {
      tmp += MATRIX_IDX(A, gx, idx_k, M) * MATRIX_IDX(B, idx_k, gy, K);
    }

    MATRIX_IDX(C, gx, gy, M) = alpha * tmp + beta * MATRIX_IDX(C, gx, gy, M);
  }
}