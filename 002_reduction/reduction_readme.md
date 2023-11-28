[medium, Parallel Reduction with CUDA](https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c)
[github, cuda-reduction-example](https://github.com/umfranzw/cuda-reduction-example), has slides inside

## reduce v0

基本算法结构分为两层，一层在 host 侧，一层在 device 侧，device 侧只做 block 内 reduce，host 侧循环调用 device reduce，直到 reduce 到一个值。

| logn | MFLOPs |
| ---- | -----: |
| 11   |     21 |
| 12   |     35 |
| 21   |   2885 |
| 22   |   2261 |
| 23   |   2383 |


### host 侧 kernel launch 调用

kernel launch 的次数 $log_{2 * max_block_threads}(n)$，即 2^11, 2^22, 2^33 分别只需要 launch 1，2，3 次 kernel。

需要 n/2 个 thread 做 reduce，block 的数量如何设定：

```cpp
// 1024 是 block 中 thread 最大数量
blocks = threads_needed / 1024 + (threads_needed % 1024 ? 1 : 0);
```

block_threads 可以固定为 1024，即 cuda 的最多线程数量。在 kernel 内部决定哪些线程做那些事。

```cpp
cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
```

### reduce kernel

输入一个结果，输出到紧排列的 output buffer，input 也被改了。一个 thread block 作用域 block size * 2 个数，即 chunk size = block size * 2。

```cpp
// 每个 block 会作用到 2 * block_size 的数据
for (stride = 1; stride < chunk_size; stride *= 2, threads /= 2) {
    left = block_start + id * (stride * 2)
    right = left + stride

    // id < threads 是强要求，只能让前部分 thread 工作
    if (id < threads && right < n) {
        input[left] += input[right]
    }
    __syncthreads()
}
```

### Summary

- 为何要多次 launch kernel，不直接在 kernel 中完成 reduce 工作？
kernel 内部不能在 block 之间做 reduce，所以需要返回 host 侧，从而实现 block 之间同步。