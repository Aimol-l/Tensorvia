// Start/stop slice of a strided input into a contiguous output. Every
// dimension carries a start offset (zero for dimensions that are not sliced).
#include "common.cuh"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "interface/backend/cuda/cuda_context.hpp"
#include "interface/backend/cuda/cuda_ops.hpp"
#include "interface/backend/cuda/cuda_tensor.hpp"
#include "interface/ops.hpp"
#include "interface/tensor.hpp"
#include "interface/types.hpp"

#include "cuda_launch.hpp"

namespace tvv::cuda {
namespace {

template <class T>
__global__ void slice_kernel(const T* in, T* out, Slice p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        int64_t rem = i;
        int64_t in_off = p.offset;
        for (int d = p.ndim - 1; d >= 0; --d) {
            const int64_t dim = p.out_shape[d];
            const int64_t c = dim > 0 ? rem % dim : 0;
            if (dim > 0) rem /= dim;
            in_off += (c + p.starts[d]) * p.in_strides[d];
        }
        out[i] = in[in_off];
    }
}

}  // namespace

cudaError_t launch_slice(cudaStream_t stream, const void* in, void* out,
                         DataType dtype, const Slice& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (p.ndim <= 0 || p.ndim > kMaxDims) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        slice_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

namespace ops {

Tensor SliceImpl<Device::CUDA>::execute(
    const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const int ndim = static_cast<int>(t.shape().size());
    if (ndim > tvv::cuda::kMaxDims) throw std::runtime_error("cuda Slice: too many dimensions");
    if (static_cast<int>(ranges.size()) != ndim)
        throw std::runtime_error("Slice ranges size must be equal to tensor shape size");

    std::vector<int64_t> out_shape(static_cast<size_t>(ndim));
    for (int i = 0; i < ndim; ++i) out_shape[i] = ranges[i].second - ranges[i].first;

    Tensor out(out_shape, t.dtype(), Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl) throw std::runtime_error("cuda Slice: failed to allocate the result");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(out_impl->context());
    if (!ctx) throw std::runtime_error("cuda Slice: missing CUDAContext");

    tvv::cuda::Slice p{};
    p.numel = static_cast<int64_t>(out.numel());
    // t.data() already points at the view origin; the kernel's offset stays 0.
    p.offset = 0;
    p.ndim = ndim;
    for (int d = 0; d < ndim; ++d) {
        p.in_shape[d] = t.shape(d);
        p.out_shape[d] = out_shape[d];
        p.in_strides[d] = t.strides(d);
        p.starts[d] = ranges[d].first;
    }

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_slice(static_cast<cudaStream_t>(ctx->stream()), t.data(),
                                out.data(), t.dtype(), p),
        "Slice launch");
    ctx->wait();
    return out;
}

}  // namespace ops
