// Device::CUDA 的 2-D 池化（Max / Avg）。
//
// 一个输出元素 = 一个线程的一次窗口扫描：池化是带宽型操作（每个输出只读
// kh*kw 个输入，无复用），一线程一输出即够，不上 shared memory tiling——
// conv.cu 的一线一输出路径是同一思路。半精度在 promoted_t<T>（float）里
// 累加后写回；Max 的哨兵用首值初始化而非 numeric_limits::lowest()（宿主
// constexpr 函数进不了 __global__，见 common.cuh 的注释）。
#include "common.cuh"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;

template <class T, bool kAvg>
__global__ void pool2d_kernel(const T* in, T* out, ops::Pool2dShape p) {
    using Acc = tvv::cuda::contract::promoted_t<T>;

    // 不用 Pool2dShape::numel()：它是宿主成员函数，设备代码里调不到。
    const int64_t total = p.batch * p.channels * p.out_h * p.out_w;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t ox = idx % p.out_w;
    int64_t rest     = idx / p.out_w;
    const int64_t oy = rest % p.out_h;
    rest /= p.out_h;
    const int64_t c  = rest % p.channels;
    const int64_t n  = rest / p.channels;

    const int64_t in_plane  = p.in_h * p.in_w;
    const int64_t out_plane = p.out_h * p.out_w;
    const T* in_plane_ptr = in + (n * p.channels + c) * in_plane;

    Acc acc    = Acc(0);
    bool first = true;
    int64_t count = 0;
    for (int64_t i = 0; i < p.kh; ++i) {
        const int64_t y = oy * p.stride_h + i - p.padding;
        if (y < 0 || y >= p.in_h) continue;
        const T* in_row = in_plane_ptr + y * p.in_w;
        for (int64_t j = 0; j < p.kw; ++j) {
            const int64_t x = ox * p.stride_w + j - p.padding;
            if (x < 0 || x >= p.in_w) continue;
            const Acc v = tvv::cuda::to_compute<Acc>(in_row[x]);
            if constexpr (kAvg) {
                acc += v;
            } else if (first || v > acc) {
                acc  = v;
                first = false;
            }
            ++count;
        }
    }
    // 窗口整体越界（没有真值）时填 0，Avg 也免了除零。
    const Acc result = count == 0 ? Acc(0)
                                  : (kAvg ? acc / static_cast<Acc>(p.kh * p.kw) : acc);
    out[(n * p.channels + c) * out_plane + oy * p.out_w + ox] =
        tvv::cuda::store_as<T>(result);
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_pool2d(cudaStream_t stream, const void* in, void* out, DataType dtype,
                          const ops::Pool2dShape& p, bool average) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        // numel() 是宿主成员函数，这里按公式重算一遍。
        const int64_t total = p.batch * p.channels * p.out_h * p.out_w;
        if (average)
            pool2d_kernel<T, true><<<blocks_for(total), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), p);
        else
            pool2d_kernel<T, false><<<blocks_for(total), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace ops {

namespace {

Tensor pool2d_execute(const Tensor& input, const Pool2dShape& shape, bool average) {
    // 公共入口已经把操作数整理成连续布局；这里只兜住绕过入口的调用。
    const Tensor src = input.is_contiguous() ? input : input.clone();

    Tensor result({shape.batch, shape.channels, shape.out_h, shape.out_w}, src.dtype(),
                  Device::CUDA);

    const auto in_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!in_impl)
        throw std::runtime_error("cuda pool2d: operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(in_impl->context());
    if (!ctx) throw std::runtime_error("cuda pool2d: missing CUDAContext");

    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!out_impl) throw std::runtime_error("cuda pool2d: failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        ::tvv::cuda::launch_pool2d(static_cast<cudaStream_t>(ctx->stream()),
                                   in_impl->data(), out_impl->data(), src.dtype(), shape,
                                   average),
        "pool2d launch");
    return result;
}

}  // namespace

Tensor MaxPool2dImpl<Device::CUDA>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, false);
}

Tensor AvgPool2dImpl<Device::CUDA>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, true);
}

}  // namespace ops
