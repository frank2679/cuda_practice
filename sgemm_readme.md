当前实现用的 Vcore，其实还可以用 cublasLt 来用 vcore 加速。

[SGEMM Implementation and Optimization](https://github.com/Huanghongru/SGEMM-Implementation-and-Optimization)
[Fast CUDA matrix multiplication from scratch](https://github.com/siboehm/SGEMM_CUDA)
[NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)


# Optimize Summary


1. 使用 shared memory
2. 一维 thread tile 并行优化
3. 二维 thread tile 并行优化
4. 寄存器缓存
5. float4 向量访存
6. 双缓存预取


## Hardware model

- 异构核，Vcore，Tcore
- Vcore Tcore 之间通过什么传递数据

## 概念对标 BR vs NV

SM vs SPC 



## Todo

1. 对比 tcore 版本，用上 tcore，跑 cutlass
2. 对标 NV vs BR 概念

# 过程

## naive impl

计算访存比，1/2

## use shared memory

BM, BN, BK = 32, 用上 shared memory，每个 block 从 global memory 上访问 (K/BK)*(BM*BK+BK*BN) 个数

访存减少到 1/32

## 一维 tile 优化

进一步减少访存比
