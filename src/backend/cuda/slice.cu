#include "backend/cuda/ops/slice.h"

namespace ops {

template <typename T, int MAX_DIM = 4>
__global__ void slice_cuda(
    T* output_data,
    const T* input_data,
    size_t* input_strides,
    int* input_shape,
    int* slice_starts,
    int* output_shape,
    size_t input_dims,
    size_t slice_dim,
    size_t output_dims,
    size_t total_elements) {
    // 计算全局索引
    size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear_idx >= total_elements)
        return;

    // 将线性索引展开成多维坐标
    size_t indices[MAX_DIM];
    size_t remaining = linear_idx;

    for (int i = output_dims - 1; i >= 0; --i) {
        indices[i] = remaining % output_shape[i];
        remaining /= output_shape[i];
    }

    // 计算在原始张量中的偏移
    size_t src_offset = 0;
    for (size_t i = 0; i < input_dims; ++i) {
        int coord;
        if (i < slice_dim) {
            coord = slice_starts[i] + indices[i];
        } else {
            coord = indices[i];
        }
        src_offset += coord * input_strides[i];
    }

    // 复制数据
    output_data[linear_idx] = input_data[src_offset];
}

Tensor SliceImpl<Device::CUDA>::execute(const Tensor& a, const std::vector<std::pair<int, int>>& ranges) {
    auto a_shape = a.shape();
    size_t a_dim = a_shape.size();
    const size_t slice_dim = ranges.size();
    // 构建新 shape（切片维度为 [start, end)，非切片维度原样保留）
    std::vector<int64_t> new_shape;
    new_shape.reserve(a_dim);
    for (size_t i = 0; i < slice_dim; ++i) {
        new_shape.push_back(ranges[i].second - ranges[i].first);
    }
    for (size_t i = slice_dim; i < a_dim; ++i) {
        new_shape.push_back(a_shape[i]);
    }
    // 创建输出 Tensor
    Tensor res(new_shape, a.dtype(), a.device());

    size_t output_dim = new_shape.size();

    // 准备切片起始位置数组
    auto range_dim = ranges.size();
    std::vector<int64_t> slice_starts(range_dim);
    for (size_t i = 0; i < range_dim; ++i) {
        slice_starts[i] = ranges[i].first;
    }

    // 计算输入步长
    std::vector<size_t> strides(a_dim, 1);
    for (int i = a_dim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * a_shape[i + 1];
    }

    constexpr int threads = 256;
    int blocks = (a.numel() + threads - 1) / threads;
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(res.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());

    cudaStream_t stream = ctx_impl->stream();

    // 使用托管内存分配，后期可以考虑使用动态共享内存进行优化
    size_t* d_input_strides = nullptr;
    int* d_input_shape = nullptr;
    int* d_slice_starts = nullptr;
    int* output_shape = nullptr;

    cudaMallocManaged(&d_input_strides, a_dim * sizeof(size_t));
    cudaMallocManaged(&d_input_shape, a_dim * sizeof(int));
    cudaMallocManaged(&d_slice_starts, slice_dim * sizeof(int));
    cudaMallocManaged(&output_shape, a_dim * sizeof(int));

    // 直接复制数据到托管内存
    memcpy(d_input_strides, strides.data(), a_dim * sizeof(size_t));
    memcpy(d_input_shape, a_shape.data(), a_dim * sizeof(int));
    memcpy(d_slice_starts, slice_starts.data(), slice_dim * sizeof(int));
    memcpy(output_shape, new_shape.data(), output_dim * sizeof(int));

    std::visit([&](auto a_ptr) {  // 没有数学运算，不需要分开考虑float16和bfloat64,直接复制即可
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        slice_cuda<AType><<<blocks, threads, 0, stream>>>(static_cast<AType*>(res.data()), static_cast<const AType*>(a.data()), d_input_strides, d_input_shape, d_slice_starts, output_shape, a_dim, range_dim, output_dim, res.numel());
    },A);
    ctx_impl->wait();
    cudaFree(d_input_strides);
    cudaFree(d_input_shape);
    cudaFree(d_slice_starts);
    cudaFree(output_shape);
    return res;
}

template struct SliceImpl<Device::CUDA>;

}  // namespace ops