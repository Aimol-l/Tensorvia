#include "backend/cuda/ops/transpose.h"


namespace ops {

// template <typename T>
// __global__ void transpose_cuda2d(
//     T* input,      // 输入矩阵（行优先）
//     T* output,     // 输出矩阵（行优先）
//     int rows,      // 行数
//     int cols)      // 列数
// {
//     size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
//     size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//     if (idx >= total) return;
//     // 原矩阵坐标
//     size_t i = idx / cols; // 行
//     size_t j = idx % cols; // 列
//     // 转置后的线性索引
//     size_t t_idx = static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i);
//     // if (input == output) {
//         // 原地转置：避免重复交换
//         if (idx < t_idx) {
//             T tmp = input[idx];
//             input[idx] = input[t_idx];
//             input[t_idx] = tmp;
//         }
//     // } else {
//     //     // 非原地
//     //     output[t_idx] = input[idx];
//     // }
// }

// template <typename T>
// __global__ void transpose_cuda_nd(
//     const T* __restrict__ in,
//     T* __restrict__ out,
//     const int* __restrict__ in_strides,
//     const int* __restrict__ out_strides,
//     const int* __restrict__ axes,
//     const int* __restrict__ in_shape,
//     int ndim,
//     size_t numel
// ) {
//     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= numel) return;

//     // 计算多维坐标 old_idx
//     int coord[16]; // 假设 <= 16 维
//     size_t tmp = tid;
//     for (int i = 0; i < ndim; ++i) {
//         coord[i] = tmp / in_strides[i];
//         tmp %= in_strides[i];
//     }

//     // 生成新坐标 new_coord
//     int new_coord[16];
//     for (int i = 0; i < ndim; ++i) {
//         new_coord[i] = coord[axes[i]];
//     }

//     // 计算输出位置
//     size_t out_index = 0;
//     for (int i = 0; i < ndim; ++i) {
//         out_index += new_coord[i] * out_strides[i];
//     }

//     out[out_index] = in[tid];
// }

void TransposeImpl<Device::CUDA>::execute(Tensor& a){ 

}

Tensor TransposeImpl<Device::CUDA>::execute(Tensor& a, std::initializer_list<int> axes) { 
    return a.clone();
}

template struct TransposeImpl<Device::CUDA>;

}