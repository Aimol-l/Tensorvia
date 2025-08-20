#include "backend/cuda/ops/mul.h"

namespace ops {

// [w,k] @ [k,h] --> [w,h]
// [b,w,k] @ [b,k,h] --> [b,w,h]
template<typename T>
__global__ void mul_cuda_3d(
    T* a, T* b, T* res,
    size_t batch, size_t w, size_t h, size_t k
) {
    constexpr int TILE = 32;
    // 每个 block 负责 TILE x TILE 的输出区域
    __shared__ T tile_a[TILE][TILE];
    __shared__ T tile_b[TILE][TILE];
    // 全局线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;  // 输出列块
    int by = blockIdx.y;  // 输出行块
    int bz = blockIdx.z;  // batch 索引
    // 当前 block 负责的输出起始位置
    int row = by * TILE + ty;
    int col = bx * TILE + tx;
    // 累加器（寄存器）
    T sum = T(0);
    // 遍历 k 维度，分块加载
    for (int tile_idx = 0; tile_idx < (k + TILE - 1) / TILE; ++tile_idx) {
        // 加载 tile_a: a[b][row][tile_k]
        int a_k = tile_idx * TILE + tx;
        if (row < w && a_k < k) {
            tile_a[ty][tx] = a[bz * w * k + row * k + a_k];
        } else {
            tile_a[ty][tx] = T(0);
        }
        // 加载 tile_b: b[b][tile_k][col]
        int b_k = tile_idx * TILE + ty;
        if (b_k < k && col < w) {
            tile_b[ty][tx] = b[bz * k * w + b_k * w + col];
        } else {
            tile_b[ty][tx] = T(0);
        }
        __syncthreads();
        // 内层计算：当前 tile 的乘积累加
        for (int i = 0; i < TILE; ++i) {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }
        __syncthreads();
    }
    // 写回结果
    if (row < w && col < w) {
        res[bz * w * w + row * w + col] = sum;
    }
}

Tensor MulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return a.clone();
}

template struct MulImpl<Device::CUDA>;

}