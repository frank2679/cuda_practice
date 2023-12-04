#pragma once

#include <cuda_runtime.h>

void randomize_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void cudaCheck(cudaError_t error, const char *file, int line);
bool verify_matrix(float *matRef, float *matOut, int N);
