/* Comp 4510 - CUDA Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * - uses shared memory, check out the kernel.cu
 */

#include "kernels.h"
#include "perf.h"
#include "reduce.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // Grab n from the command line args
  const unsigned int n = parse_args(argc, argv);
  unsigned int remaining;

  // Query device to find max num threads per block
  int block_threads = get_max_block_threads();
  printf("max threads per block: %d\n", block_threads);

  // Prepare timer objects to track performance
  Perf perf = create_perf();

  // All CUDA API calls return a status, which we must check
  cudaError_t status;

  // Host buffers
  float *host_array;
  float final_sum = 0.001;
  float partial_results[STREAMS]; // this will hold the partial sum from each
                                  // stream

  // Device buffers
  float *dev_input[STREAMS];
  float *dev_output[STREAMS];

  // Size info for kernel launches
  unsigned int threads_needed = n / 2 / STREAMS;
  unsigned int blocks = threads_needed / block_threads;

  if (blocks > 0) {
    block_threads = 1024;
  } else {
    block_threads = threads_needed;
    blocks = 1;
  }

  // Allocate host buffer and fill it
  status = cudaMallocHost(&host_array, n * sizeof(float));
  check_error(status, "Error allocating host buffer.");
  init_array(host_array, n);

  // Allocate device buffers
  // Note: Input buffer needs to be of size n.
  // Output buffer is used to store partial results after each kernel launch.
  // On first launch, each block will reduce block_size * 2 values down to 1
  // value (except last kernel, which may reduce less than block_size * 2
  // values down to 1 value if  n is not a multiple of block_size * 2). On
  // subsequent launches, we need even less output buffer space. Therefore
  // output buffer size needs to be equal to the number of blocks required for
  // first launch.

  for (unsigned int i = 0; i < STREAMS; ++i) {
    status = cudaMalloc(&dev_input[i], n / STREAMS * sizeof(float));
    check_error(status, "Error allocating device buffer.");

    // had bug here, blocks for output, not need to divide STREAMS any more
    // blocks for each launch
    status = cudaMalloc(&dev_output[i], blocks * sizeof(float));
    check_error(status, "Error allocating device buffer.");
  }

  cudaStream_t streams[STREAMS];
  for (int i = 0; i < STREAMS; i++) {
    status = cudaStreamCreate(&(streams[i]));
    check_error(status, "Error creating CUDA stream.");
  }

  // Start the program timer
  start_timer(&(perf.total_timer));

  // Transfer the input array from host to device
  start_timer(&(perf.h2d_timer));
  for (int i = 0; i < STREAMS; ++i) {
#if ASYNC_API
    status = cudaMemcpyAsync(dev_input[i], host_array + i * n / STREAMS,
                             n / STREAMS * sizeof(float),
                             cudaMemcpyHostToDevice, streams[i]);
#else
    status = cudaMemcpy(dev_input[i], host_array + i * n / STREAMS,
                        n / STREAMS * sizeof(float), cudaMemcpyHostToDevice);
#endif
    check_error(status, "Error on CPU->GPU cudaMemcpy for array.");
  }

  stop_timer(&(perf.h2d_timer));

  // Launch kernel
  // Note: We call the kernel multiple times - each call reduces the size of the
  // array by 2.
  start_timer(&(perf.kernel_timer));

  remaining = n / STREAMS;
  int while_loop_idx = 0;
  while (remaining > 1) {
#if DEBUG_INFO
    printf("Launching kernels:\n");
    printf("remaining: %u\n", remaining);
    printf("blocks: %u\n", blocks);
    printf("threads_needed: %u\n", threads_needed);
    printf("block_threads: %d\n", block_threads);
    printf("\n");
#endif

    for (int i = 0; i < STREAMS; ++i) {
      // cudaStreamSynchronize(streams[i]);
      reduce_yh<<<blocks, block_threads, 0, streams[i]>>>(
          dev_input[i], dev_output[i], remaining);
#if DEBUG_INFO
      // Note: the kernel launches in the loop above are asychronous, so this
      // may
      // not necessarily catch kernel errors... If they're not caught here,
      // they'll be caught in the check_error() call after the next blocking
      // operation (the GPU -> CPU data transfer below).
      printf("while_loop_idx: %d, stream: %d\n", while_loop_idx, i);
#endif
      check_error(cudaGetLastError(), "Error launching kernel.");
    }

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
      for (int i = 0; i < STREAMS; ++i) {
        float *dev_tmp = dev_input[i];
        dev_input[i] = dev_output[i];
        dev_output[i] = dev_tmp;
      }
    }

    while_loop_idx++;
  }

  stop_timer(&(perf.kernel_timer));

  // Transfer the element in position 0 of the dev_output buffer back to the
  // host. This is the final sum.
  start_timer(&(perf.d2h_timer));

  for (int i = 0; i < STREAMS; ++i) {
#if ASYNC_API
    status = cudaMemcpyAsync(&partial_results[i], dev_output[i], sizeof(float),
                             cudaMemcpyDeviceToHost, streams[i]);
#else
    status = cudaMemcpy(&partial_results[i], dev_output[i], sizeof(float),
                        cudaMemcpyDeviceToHost);
#endif
    check_error(status, "Error on GPU->CPU cudaMemcpy for final_sum.");
  }
  stop_timer(&(perf.d2h_timer));

  // Record the final clock time
  stop_timer(&(perf.total_timer));

  // Since the GPU operates asynchronously, wait until the final time has been
  // recorded before continuing on to print the results below (this synchronizes
  // the device and the host).
  cudaEventSynchronize(perf.total_timer.stop);

  cpu_reduce(partial_results, STREAMS, &final_sum);
  printf("partial_results: %f\n", partial_results[0]);

  // Display the results & performance statistics
  print_results(n, final_sum, &perf);
  float golden_sum = 0.f;

  for (int i = 0; i < STREAMS; ++i) {
    printf("stream: %d\n", i);
    cpu_reduce(host_array + i * n / STREAMS, n / STREAMS, &golden_sum);
    check_result(golden_sum, partial_results[i], 0.000001);
  }

  // Clean up memory (both on host *AND device*!)
  status = cudaFreeHost(host_array); // Note: must use this function instead of
                                     // free() to free a *pinned* buffer
  check_error(status, "Error freeing host buffer.");
  destroy_perf(&perf);
  for (int i = 0; i < STREAMS; ++i) {
    status = cudaStreamDestroy(
        streams[i]); // Note: have to destroy the stream structs too...
    check_error(status, "Error destroying stream.");
    status = cudaFree(dev_input[i]);
    check_error(status, "Error calling cudaFree on device buffer.");
    status = cudaFree(dev_output[i]);
    check_error(status, "Error calling cudaFree on device buffer.");
  }

  printf("\n==================\n");
#if ASYNC_API
  printf("use async API\n");
#else
  printf("use sync API\n");
#endif

#if USE_RAND_VALS
  printf("use random init\n");
#else
  printf("use non-random init\n");
#endif

  printf("streams: %d\n", STREAMS);

  return EXIT_SUCCESS;
}
