#include <cuda_runtime.h>

// col-major
#define MATRIX_IDX(buf, i, j, ld) (buf + i * ld + j)

template <int BLOCKSIZE>
__global__ void gemm_kernel(float *A, float *B, float *C, int m, int n, int k)
{
    int gx = threadIdx.x + blockIdx.x * blockDim.x;
    int gy = threadIdx.y + blockIdx.y * blockDim.y;

    for (int idx_k = 0; idx_k < k; ++idx_k)
    {
        MATRIX_IDX(C, gx, gy, m) += matix(A, gx, idx_k, m) * matrix(B, dix_k, gy, k);
    }
}

int main()
{
    dim3 block(32, 32);
    dim3 grid(Ceil(M, 32), Ceil(N, 32));

    // alloc memory on device, A, B, C

    gemm_kernel<32><<<grid, block>>>(A, B, C, M, N, K);
}
