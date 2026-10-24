// Element-wise dtype conversion, plus the host driver for TypecastImpl.
#include "common.cuh"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace tvv::cuda {
namespace {

template <class T, class R>
__global__ void cast_kernel(const T* in, R* out, int64_t numel) {
    using C = contract::promoted_t<T>;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        out[i] = store_as<R>(to_compute<C>(in[i]));
    }
}

}  // namespace

cudaError_t launch_cast(cudaStream_t stream, const void* in, void* out,
                        int64_t numel, DataType in_dtype, DataType out_dtype) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using R = typename decltype(dst_tag)::type;
            cast_kernel<T, R><<<blocks_for(numel), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<R*>(out), numel);
            status = launch_status();
        });
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

namespace ops {

namespace {

Tensor cast_tensor(const Tensor& a, DataType dst_type) {
    // The kernel reads a flat buffer; materialize any strided view.
    const Tensor src = a.is_contiguous() ? a : a.clone();
    Tensor out(src.shape(), dst_type, Device::CUDA);

    const auto src_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!src_impl || !out_impl)
        throw std::runtime_error("cuda Typecast: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    if (!ctx) throw std::runtime_error("cuda Typecast: missing CUDAContext");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_cast(static_cast<cudaStream_t>(ctx->stream()), src_impl->data(),
                               out_impl->data(), static_cast<int64_t>(src.numel()),
                               src.dtype(), dst_type),
        "Typecast launch");
    ctx->wait();
    return out;
}

}  // namespace

void TypecastImpl<Device::CUDA>::execute(Tensor& a, DataType dst_type) {
    if (a.dtype() == dst_type) return;
    Tensor converted = cast_tensor(a, dst_type);
    a.set_impl(converted.get_impl(), dst_type);
}

Tensor TypecastImpl<Device::CUDA>::execute(const Tensor& a, DataType dst_type) {
    if (a.dtype() == dst_type) return a;
    return cast_tensor(a, dst_type);
}

}  // namespace ops
