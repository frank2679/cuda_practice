/* Comp 4510 - CUDA Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * - uses pinned memory to accelerate host to device transfer (see below)
 */

#include "kernels.h"
#include "perf.h"
#include "reduce.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// assume dev_input, dev_output are allocated
void reduce_yh_host(float *dev_input, float *dev_output, int n) {
  unsigned int remaining = n;
  unsigned int threads_needed = n / 2;
  unsigned int blocks = threads_needed / 1024;
  unsigned int block_threads;

  if (blocks > 0) {
    block_threads = 1024;
  } else {
    block_threads = threads_needed;
    blocks = 1;
  }

  while (remaining > 1) {
#if DEBUG_INFO
    printf("Launching kernels:\n");
    printf("remaining: %u\n", remaining);
    printf("blocks: %u\n", blocks);
    printf("threads_needed: %u\n", threads_needed);
    printf("block_threads: %u\n", block_threads);
    printf("\n");
#endif
    reduce_yh<<<blocks, block_threads>>>(dev_input, dev_output, remaining);
    remaining = blocks;
    threads_needed = remaining / 2;
    blocks = threads_needed / 1024;
    if (blocks > 0) {
      block_threads = 1024;
    } else {
      block_threads = threads_needed;
      blocks = 1;
    }
    if (remaining > 1) {
      float *dev_tmp = dev_input;
      dev_input = dev_output;
      dev_output = dev_tmp;
    }
    // cudaDeviceSynchronize();
  }
}

void reduce_host(float *dev_input, float *dev_output, int n) {
  unsigned int remaining = n; // tracks number of elements left to add
  // Size info for kernel launches
  unsigned int threads_needed;
  unsigned int blocks;
  const int block_threads = get_max_block_threads();

  float *dev_temp;        // temp pointer used to flip buffers between kernel
                          // launches (see loop below)
  threads_needed = n / 2; // we'll need one thread to add every 2 elements
  blocks = threads_needed / block_threads + \ // we'll need this many blocks
           (threads_needed % block_threads > 0
                ? 1
                : 0); // plus one extra if threads_needed
                      // does not evenly divide block_threads
  while (remaining >
         1) // continue until we have a single value left (the final sum)
  {
#if DEBUG_INFO
    printf("Launching kernels:\n");
    printf("remaining: %u\n", remaining);
    printf("blocks: %u\n", blocks);
    printf("threads_needed: %u\n", threads_needed);
    printf("\n");
#endif

    // call the kernel
    // reduce<<<blocks, block_threads>>>(dev_input, dev_output, remaining);
    reduce<<<blocks, block_threads>>>(dev_input, dev_output, remaining);

    // re-compute our size information for the next iteration
    remaining = blocks; // After the previous kernel call, each block has
                        // reduced its chunk down to a single partial sum
    threads_needed = remaining / 2; // each thread added 2 elements
    blocks = threads_needed / block_threads +
             (threads_needed % block_threads
                  ? 1
                  : 0); // again, might need one extra block if threads_needed
                        // is not evenly divisible by block_threads

    // if we will need to do another iteration, flip (swap) the device input and
    // output buffers; i.e. the output buffer from the last call becomes input
    // buffer for the next call, and the input buffer from last call is re-used
    // to store output for the next call. Note: no data is transferred back to
    // the host here, this is just a pointer operation
    if (remaining > 1) {
      dev_temp = dev_input;
      dev_input = dev_output;
      dev_output = dev_temp;
    }
  }
}

int main(int argc, char *argv[]) {
  // Grab n from the command line args
  const unsigned int n = parse_args(argc, argv);

  // Query device to find max num threads per block
  const int block_threads = get_max_block_threads();
  printf("max threads per block: %d\n", block_threads);

  // Prepare timer objects to track performance
  Perf perf = create_perf();

  // All CUDA API calls return a status, which we must check
  cudaError_t status;

  // Host buffers
  float *host_array;
  float final_sum = 0.001;

  // Device buffers
  float *dev_input;
  float *dev_output;

  // Size info for kernel launches
  unsigned int threads_needed;
  unsigned int blocks;
  unsigned int remaining; // number of elements left to add

  // Allocate host buffer and fill it
  status = cudaMallocHost(&host_array, n * sizeof(float));
  check_error(status, "Error allocating host buffer.");
  init_array(host_array, n);

  // Allocate device buffers
  // Note: Input buffer needs to be of size n.
  // Output buffer is used to store partial results after each kernel launch. On
  // first launch, each block will reduce block_size * 2 values down to 1 value
  // (except last kernel, which may reduce less than block_size * 2 values down
  // to 1 value if  n is not a multiple of block_size * 2). On subsequent
  // launches, we need even less output buffer space. Therefore output buffer
  // size needs to be equal to the number of blocks required for first launch.

  status = cudaMalloc(&dev_input, n * sizeof(float));
  check_error(status, "Error allocating device buffer.");
  status = cudaMalloc(&dev_output, blocks * sizeof(float));
  check_error(status, "Error allocating device buffer.");

  // Start the program timer
  start_timer(&(perf.total_timer));

  // Transfer the input array from host to device
  start_timer(&(perf.h2d_timer));
  status = cudaMemcpy(dev_input, host_array, n * sizeof(float),
                      cudaMemcpyHostToDevice);
  stop_timer(&(perf.h2d_timer));
  check_error(status, "Error on CPU->GPU cudaMemcpy for host_array.");

  // Launch kernel
  // Note: We call the kernel multiple times - each call reduces the size of the
  // array by 2.
  start_timer(&(perf.kernel_timer));
#if USE_YH_IMPL
  reduce_yh_host(dev_input, dev_output, n);
#else
  reduce_host(dev_input, dev_output, n);
#endif
  stop_timer(&(perf.kernel_timer));
  // Note: the kernel launches in the loop above are asychronous, so this may
  // not necessarily catch kernel errors... If they're not caught here, they'll
  // be caught in the check_error() call after the next blocking operation (the
  // GPU -> CPU data transfer below).
  check_error(cudaGetLastError(), "Error launching kernel.");

  // Transfer the element in position 0 of the dev_output buffer back to the
  // host. This is the final sum.
  start_timer(&(perf.d2h_timer));
  status =
      cudaMemcpy(&final_sum, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
  stop_timer(&(perf.d2h_timer));
  check_error(status, "Error on GPU->CPU cudaMemcpy for final_sum.");

  // Record the final clock time
  stop_timer(&(perf.total_timer));
  // Since the GPU operates asynchronously, wait until the final time has been
  // recorded before continuing on to print the results below (this synchronizes
  // the device and the host).
  cudaEventSynchronize(perf.total_timer.stop);

  // Display the results & performance statistics
  print_results(n, final_sum, &perf);
  float golden_sum = 0.f;
  cpu_reduce(host_array, n, &golden_sum);
  check_result(golden_sum, final_sum);

  // Clean up memory (both on host *AND device*!)
  status = cudaFreeHost(host_array); // Note: must use this function instead of
                                     // free() to free a *pinned* buffer
  check_error(status, "Error freeing host buffer.");
  destroy_perf(&perf);
  status = cudaFree(dev_input);
  check_error(status, "Error calling cudaFree on device buffer.");
  status = cudaFree(dev_output);
  check_error(status, "Error calling cudaFree on device buffer.");

  return EXIT_SUCCESS;
}
