[medium, Parallel Reduction with CUDA](https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c)
[github, cuda-reduction-example](https://github.com/umfranzw/cuda-reduction-example), has slides inside

## reduce v0

reduce function 处理的问题是将一个 block 中的结果 reduce 到一个值。此时 remaining 为 block 数，等到第二次 launch kernel 再进一步 reduce，直到 remaining 只剩一个值。

### reduce kernel

输入一个结果，输出到紧排列的 output buffer，input 也被改了。一个 thread block 作用域 block size * 2 个数，即 chunk size = block size * 2。

for (stride = 1; stride < chunk_size; stride *= 2, threads /= 2) {
    left = block_start + id * (stride * 2)
    right = left + stride

    // id < threads 是强要求，只能让前部分 thread 工作
    if (id < threads && right < n) {
        input[left] += input[right]
    }
    __syncthreads()
}