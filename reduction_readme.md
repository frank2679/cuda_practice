- [medium, Parallel Reduction with CUDA](https://shreeraman-ak.medium.com/parallel-reduction-with-cuda-d0ae10c1ae2c)
- [github, cuda-reduction-example](https://github.com/umfranzw/cuda-reduction-example), has slides inside


# summary

1. 算法，树状实现


# 实现

## v0 多次 launch

每次 launch 只算 block (2048, 一个 thread 加两个值，最多 1024 thread/block) 以内的和，然后紧密放在一起。循环去 launch 直到 n = 1

每次计算相邻两个值

## v1 紧排序

将每次计算输入是stride，输出都紧排序，即 coalescing

## v2 使用 pinned memory

申请 host 内存使用 cudaMallocHost

## v3 overlap memcpy and reduce

使用多个 stream 来 overlap 

## v4 reduce use shared memory

中间结果放在 shared memory 上
