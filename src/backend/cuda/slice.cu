#include "backend/cuda/ops/slice.h"


namespace ops {

// template <typename T>
// __global__ void slice_cuda(Tensor& output, const Tensor& input, const std::vector<std::pair<int, int>>& ranges) {
//     // 放在寄存器上，这里可以用最快的速度获取
//     auto t_shape = input.shape();
//     size_t t_dim = t_shape.size();
//     auto new_shape = output.shape();
//     const size_t slice_dim = ranges.size();
//     // 构建原始张量的 strides（行主序），放在共享内存上，方便其他线程获取，不重复计算
//     __shared__ size_t strides[t_dim];
//     // 只让一个线程计算，然后同步
//     if (threadIdx.x == 0) {
//         for (int i = 0; i < static_cast<int>(t_dim); ++i) {
//             strides[i] = 1;
//         }
//         for (int i = static_cast<int>(t_dim) - 2; i >= 0; --i) {
//             strides[i] = strides[i + 1] * t_shape[i + 1];
//         }
//     }
//     __syncthreads();
//     // 获取数据指针
//     const T* src_data = static_cast<const T*>(input.data());
//     T* dest_data = static_cast<T*>(output.data());

//     size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x; // 全局索引
//     if (linear_idx >= input.numel()) {
//         return;
//     }
//     // 将线性 idx 展开成多维坐标
//     size_t indices[new_shape.size()];
//     std::memset(indices, 0, sizeof(indices));
//     size_t remaining = linear_idx;
//     #pragma unroll
//     for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
//         indices[i] = remaining % new_shape[i];
//         remaining /= new_shape[i];
//     }
//     // 计算在原始张量中的偏移
//     size_t src_offset = 0;
//     #pragma unroll
//     for (size_t i = 0; i < t_dim; ++i) {
//         int coord = (i < slice_dim) ? (ranges[i].first + indices[i]) : indices[i];
//         src_offset += coord * strides[i];
//     }
//     dest_data[linear_idx] = src_data[src_offset];
    
// }


Tensor SliceImpl<Device::CUDA>::execute(const Tensor& a, const std::vector<std::pair<int, int>>& ranges) {
    return a.clone();
}

template struct SliceImpl<Device::CUDA>;


}