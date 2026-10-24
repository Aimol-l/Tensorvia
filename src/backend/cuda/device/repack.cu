// Strided-view repack: materialize an arbitrary (offset, strides) view into a
// contiguous buffer. Used by Tensor::clone / contiguous / to_host.
#include "common.cuh"
#include "cuda_launch.hpp"

#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

namespace {

__global__ void repack_kernel(const char* __restrict__ src, char* __restrict__ dst,
                              tvv::cuda::Repack p) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t idx = start; idx < p.numel; idx += stride) {
        int64_t temp = idx;
        int64_t src_offset = p.offset;
        for (int i = p.ndim - 1; i >= 0; --i) {
            const int64_t coord = temp % p.shape[i];
            temp /= p.shape[i];
            src_offset += coord * p.strides[i];
        }
        const char* from = src + src_offset * p.dtype_size;
        char* to = dst + idx * p.dtype_size;
        for (int b = 0; b < p.dtype_size; ++b) to[b] = from[b];
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_repack(cudaStream_t stream, const void* in, void* out,
                          const Metadata& meta) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (meta.shape.size() > static_cast<size_t>(kMaxDims)) return cudaErrorInvalidValue;
    if (meta.shape.size() != meta.strides.size()) return cudaErrorInvalidValue;
    if (meta.numel == 0) return cudaSuccess;

    Repack p{};
    p.numel      = static_cast<int64_t>(meta.numel);
    p.offset     = meta.offset;
    p.ndim       = static_cast<int32_t>(meta.shape.size());
    p.dtype_size = static_cast<int32_t>(calc_dtype_size(meta.dtype));
    for (size_t i = 0; i < meta.shape.size(); ++i) {
        p.shape[i]   = meta.shape[i];
        p.strides[i] = meta.strides[i];
    }

    repack_kernel<<<blocks_for(p.numel), kThreads, 0, stream>>>(
        static_cast<const char*>(in), static_cast<char*>(out), p);
    return launch_status();
}

}  // namespace tvv::cuda
