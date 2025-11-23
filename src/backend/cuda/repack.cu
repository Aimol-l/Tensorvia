#include "backend/cuda/ops/repack.h"

// CUDA kernel: 将任意布局张量拷贝为连续
__global__ void repack_kernel(
    const char* __restrict__ src, char* __restrict__ dst,
    size_t numel,
    const int64_t* shape, const int64_t* strides,
    size_t ndim,
    size_t dtype_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    // 将线性索引 idx 转为多维坐标
    size_t temp = idx;
    size_t src_offset = 0;
    for (int i = ndim - 1; i >= 0; --i) {
        size_t coord = temp % shape[i];
        temp /= shape[i];
        src_offset += coord * strides[i];
    }
    // 拷贝 dtype_size 字节
    for (size_t b = 0; b < dtype_size; ++b) {
        dst[idx * dtype_size + b] = src[src_offset * dtype_size + b];
    }
}

void RepackImpl<Device::CUDA>::execute(const Metadata& meta,void* input,void* output) {

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape,   meta.shape.size() * sizeof(int64_t));
    cudaMalloc(&d_strides, meta.shape.size() * sizeof(int64_t));
    cudaMemcpy(d_shape,   meta.shape.data(),   meta.shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, meta.strides.data(), meta.shape.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    // 启动 kernel
    const char* src_base = static_cast<const char*>(input) + meta.offset * calc_dtype_size(meta.dtype);
    char* dst = static_cast<char*>(output);

    size_t block_size = 256;
    size_t grid_size = (meta.numel + block_size - 1) / block_size;

    repack_kernel<<<grid_size, block_size>>>(
        src_base, dst, meta.numel,
        d_shape, d_strides,
        meta.shape.size(),
        calc_dtype_size(meta.dtype)
    );

    cudaFree(d_shape);
    cudaFree(d_strides);
    cudaDeviceSynchronize();
}


template struct RepackImpl<Device::CUDA>;
