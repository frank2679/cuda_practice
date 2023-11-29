/* Some misc. utility functions for things like error checking, array generation, and device querying.
  */

#include "reduce.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

// Checks if an error occurred using the given status.
// If so, prints the given message and halts.
void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately before the rest of the train derails...
    }
}

// Returns the maximum number of supported threads per block on the current device.
// Note: This is a hardware-enforced limit.
int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    // Grab the device number of the default CUDA device.
    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    // Query the max possible number of threads per block
    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}

// Fills an array with random floats in the range [0, 1],
// or the value 1.0 (depending on USE_RAND_VALS flag)
void init_array(float *array, int len)
{
    srand(time(NULL));
    
    int i;
    for (i = 0; i < len; i++)
    {
        // keep the values small to avoid overflow during the summation
#if USE_RAND_VALS
        array[i] = (float) rand() / RAND_MAX;
#else
        array[i] = 1.0;
#endif
    }    
}

// Prints the given array to stdout (for debugging)
void print_array(const char *label, float *array, unsigned int len)
{
    printf("%s", label);
    
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n\n");
}

// Reads the value of i from the command line array and returns n = 2^i
unsigned int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./reduce <i, where n = 2^i>\n");
        exit(1);
    }

    // return (unsigned int)atoi(argv[1]);
    return (unsigned int) pow(2, atoi(argv[1]));
}

// Kahan summation algorithm to resolve numerical error
// https://sss.cs.vt.edu/pubs/pldi03.pdf
/* The Kahan summation algorithm can help reduce the numerical instability in
floating-point summation by keeping track of the lost precision and adding it
back to the sum. The algorithm works by using a separate variable to accumulate
the lost precision, which is then added back to the sum in the next iteration1.
This helps to reduce the error that accumulates due to the limited precision of
floating-point numbers.

The Kahan summation algorithm is more accurate than the naive approach of simply
summing the numbers, especially when summing a large number of values1. However,
it does come with a performance cost due to the additional operations required
to keep track of the lost precision1.
*/
void cpu_reduce(float *array, int len, float *asum) {
  float sum = 0.0;
  float c = 0.0;
  for (int i = 0; i < len; i++) {
    float y = array[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  *asum = sum;
}

bool check_result(float golden, float output, float threshold) {
  float diff = fabs(golden - output);
  bool status = false;
  if (diff > threshold) {
    printf("Case Failed, golden: %f, output: %f\n", golden, output);
  } else {
    printf("Case passed. \n");
    status = true;
  }
  printf("The difference is %f, relative diff: %f\n", diff, diff / golden);
  return status;
}