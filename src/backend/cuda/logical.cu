#include "ops.h"
#include "backend/cuda/ops/logical.h"

namespace ops {

// template <typename T>
// __global__ void equal_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         if constexpr (std::is_floating_point_v<T>) {
//             out_ptr_[i] = (fabs(a_ptr_[i] - b_ptr_[i]) < 1e-6f) ? 1 : 0;
//         } else {
//             out_ptr_[i] = (a_ptr_[i] == b_ptr_[i]) ? 1 : 0;
//         }
//     }
// }
// template <typename T>
// __global__ void not_equal_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         if constexpr (std::is_floating_point_v<T>) {
//             out_ptr_[i] = (fabs(a_ptr_[i] - b_ptr_[i]) >= 1e-6f) ? 1 : 0;
//         } else {
//             out_ptr_[i] = (a_ptr_[i] != b_ptr_[i]) ? 1 : 0;
//         }
//     }
// }
// template <typename T>
// __global__ void greater_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         out_ptr_[i] = (a_ptr_[i] > b_ptr_[i]) ? 1 : 0;
//     }
// }
// template <typename T>
// __global__ void less_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         out_ptr_[i] = (a_ptr_[i] < b_ptr_[i]) ? 1 : 0;
//     }
// }
// template <typename T>
// __global__ void greater_equal_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         out_ptr_[i] = (a_ptr_[i] >= b_ptr_[i]) ? 1 : 0;
//     }
// }
// template <typename T>
// __global__ void less_equal_cuda(const T* a_ptr_, const T* b_ptr_, int8_t* out_ptr_, size_t size) {
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < size) {
//         out_ptr_[i] = (a_ptr_[i] <= b_ptr_[i]) ? 1 : 0;
//     }
// }
// template <typename T>
// __global__ void non_zero_cuda(const T* a_ptr, size_t size, size_t* result) {
//      __shared__ uint32_t sdata[];  // 使用 uint32_t 更高效
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

//     // 每个线程统计自己负责的元素是否非零
//     sdata[tid] = (i < size && a_ptr[i] != T(0)) ? 1 : 0;
//     __syncthreads();

//     // 并行规约求和
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }

//     // 第一个线程将块的结果原子加到全局结果
//     if (tid == 0) {
//         atomicAdd(result, sdata[0]);
//     }
// }
Tensor EqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) { 
    return a.clone();
}

Tensor NotEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    return a.clone();
}

Tensor GreaterImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    return a.clone();
}
Tensor LessImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    return a.clone();
}
Tensor GreaterEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    return a.clone();
}
Tensor LessEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    return a.clone();
}
size_t NonZeroImpl<Device::CUDA>::execute(const Tensor& a) {
    return 0;
}
template struct EqualImpl<Device::CUDA>;
template struct NotEqualImpl<Device::CUDA>;
template struct GreaterImpl<Device::CUDA>;
template struct LessImpl<Device::CUDA>;
template struct GreaterEqualImpl<Device::CUDA>;
template struct LessEqualImpl<Device::CUDA>;
template struct NonZeroImpl<Device::CUDA>;
}