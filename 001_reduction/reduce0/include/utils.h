#ifndef _UTILS_H
#define _UTILS_H

#include "time.h"

void check_error(cudaError_t status, const char *msg);
int get_max_block_threads();
void init_array(float *array, int len);
void print_array(const char *label, float *array, unsigned int len);
unsigned int parse_args(int argc, char *argv[]);
void cpu_reduce(float *array, int len, float *sum);
bool check_result(float golden, float output, float threshold = 0.001);

#endif
