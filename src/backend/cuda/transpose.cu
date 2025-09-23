#include "backend/cuda/ops/transpose.h"


namespace ops {

template <typename T>
__global__ void transpose_cuda2d(
    T* input,      // 输入矩阵（行优先）
    T* output,     // 输出矩阵（行优先）
    int rows,      // 行数
    int cols)      // 列数
{
    size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // 原矩阵坐标
    size_t i = idx / cols; // 行
    size_t j = idx % cols; // 列
    // 转置后的线性索引
    size_t t_idx = static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i);
    // if (input == output) {
        // 原地转置：避免重复交换
        if (idx < t_idx) {
            T tmp = input[idx];
            input[idx] = input[t_idx];
            input[t_idx] = tmp;
        }
    // } else {
    //     // 非原地
    //     output[t_idx] = input[idx];
    // }
}


void TransposeImpl<Device::CUDA>::execute(Tensor& a){ 
    // 这个只支持二维转置
    auto a_shape = a.shape();
    int rows = a_shape[0];
    int cols = a_shape[1];
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());

    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        transpose_cuda2d<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<AType*>(a.data()), static_cast<AType*>(a.data()), rows, cols);
    },A);
    ctx_impl->wait();
    std::vector<int64_t> shape = {a.shape(1), a.shape(0)};
    a.reshape(shape);
}

template <typename T, const int MAX_DIM = 8>
__global__ void transpose_cuda_nd(
    const T* in,
    T* out,
    const int* in_strides,
    const int* out_strides,
    const int* axes,
    const int* in_shape,
    int ndim,
    size_t numel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    // 计算多维坐标 old_idx
    int coord[MAX_DIM]; // 假设 <= 8 维
    int tmp = tid;
    for (int i = 0; i < ndim; ++i) {
        coord[i] = tmp / in_strides[i];
        tmp %= in_strides[i];
    }

    // 生成新坐标 new_coord
    int new_coord[MAX_DIM];
    for (int i = 0; i < ndim; ++i) {
        new_coord[i] = coord[axes[i]];
    }

    // 计算输出位置
    int out_index = 0;
    for (int i = 0; i < ndim; ++i) {
        out_index += new_coord[i] * out_strides[i];
    }

    out[out_index] = in[tid];
}

Tensor TransposeImpl<Device::CUDA>::execute(Tensor& a, std::initializer_list<int64_t> axes) { 
    std::vector<int64_t> new_shape;
    std::vector<int64_t> a_shape = a.shape();
    for (auto axe : axes) new_shape.push_back(a_shape[axe]);
    Tensor result(new_shape, a.dtype(), Device::CUDA);
    const int ndim = a_shape.size();
    const int numel = a.numel();
    // 计算输入和输出的步长
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * a_shape[i + 1];
        out_strides[i] = out_strides[i + 1] * result.shape(i + 1);
    }

    int* d_in_strides, * d_out_strides, * d_axes, *input_shape;
    

    cudaMallocManaged(&d_in_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_out_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_axes, sizeof(int) * ndim);
    cudaMallocManaged(&input_shape, sizeof(int) * ndim);
    memcpy(d_in_strides, in_strides.data(), sizeof(int) * ndim);
    memcpy(d_out_strides, out_strides.data(), sizeof(int) * ndim);
    memcpy(d_axes, axes.begin(), sizeof(int) * ndim);
    memcpy(input_shape, a_shape.data(), sizeof(int) * ndim);
    
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        transpose_cuda_nd<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), d_in_strides, d_out_strides, d_axes, input_shape, ndim, numel);
    },A);
    ctx_impl->wait();
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_axes);
    cudaFree(input_shape);
    return result;
    
}

template struct TransposeImpl<Device::CUDA>;

}