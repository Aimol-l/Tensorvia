// Device::CUDA 的融合块 Conv2d+Relu+MaxPool2d(2x2)。
//
// 一个线程算一个最终输出（pool 输出）元素：它的 2x2 pool 窗口对应 conv
// 输出的 4 个位置，每个位置是一次 3x3（或 kh x kw）直接卷积，relu 后取
// max。不搬 shared：每个输出只读 kh*kw*4*in_ch 个输入、无跨输出复用，
// 是带宽型；换来的是三个算子合成一次提交与一次同步。
//
// 半精度在 promoted_t<T>（float）里累加；Max 的哨兵用首值初始化而非
// numeric_limits——宿主 constexpr 函数进不了 __global__（common.cuh）。
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

template <class T>
__global__ void conv_pool_kernel(const T* in, const T* w, const T* bias, T* out,
                                 ops::ConvPoolShape p) {
    using Acc = tvv::cuda::contract::promoted_t<T>;
    const auto& c = p.conv;

    // 不用 ConvPoolShape::numel()：宿主成员函数，设备代码里调不到。
    const int64_t total = c.batch * c.out_ch * p.out_h * p.out_w;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t opx = idx % p.out_w;
    int64_t rest      = idx / p.out_w;
    const int64_t opy = rest % p.out_h;
    rest /= p.out_h;
    const int64_t oc = rest % c.out_ch;
    const int64_t n  = rest / c.out_ch;

    const int64_t in_plane  = c.in_h * c.in_w;
    const int64_t out_plane = p.out_h * p.out_w;
    const int64_t kplane    = c.kh * c.kw;

    const Acc b = bias != nullptr ? tvv::cuda::to_compute<Acc>(bias[oc]) : Acc(0);

    // pool 窗口的 2x2 个 conv 输出位置，各做一次卷积后 relu，取 max。
    Acc best  = Acc(0);
    bool first = true;
    for (int64_t di = 0; di < 2; ++di) {
        const int64_t oy = opy * 2 + di;  // pool stride 固定 2
        for (int64_t dj = 0; dj < 2; ++dj) {
            const int64_t ox = opx * 2 + dj;
            Acc acc = b;
            for (int64_t ci = 0; ci < c.in_ch; ++ci) {
                const T* in_c = in + (n * c.in_ch + ci) * in_plane;
                const T* w_c  = w + (oc * c.in_ch + ci) * kplane;
                for (int64_t i = 0; i < c.kh; ++i) {
                    const int64_t y = oy * c.stride + i - c.padding;
                    if (y < 0 || y >= c.in_h) continue;
                    const T* in_row = in_c + y * c.in_w;
                    for (int64_t j = 0; j < c.kw; ++j) {
                        const int64_t x = ox * c.stride + j - c.padding;
                        if (x < 0 || x >= c.in_w) continue;
                        acc += tvv::cuda::to_compute<Acc>(in_row[x]) *
                               tvv::cuda::to_compute<Acc>(w_c[i * c.kw + j]);
                    }
                }
            }
            const Acc v = acc > Acc(0) ? acc : Acc(0);  // relu
            if (first || v > best) {
                best  = v;
                first = false;
            }
        }
    }
    out[(n * c.out_ch + oc) * out_plane + opy * p.out_w + opx] =
        tvv::cuda::store_as<T>(best);
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_conv_pool(cudaStream_t stream, const void* in, const void* weight,
                             const void* bias, void* out, DataType dtype,
                             const ops::ConvPoolShape& p) {
    if (in == nullptr || weight == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const int64_t total = p.conv.batch * p.conv.out_ch * p.out_h * p.out_w;
        conv_pool_kernel<T><<<blocks_for(total), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<const T*>(weight),
            static_cast<const T*>(bias), static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace ops {

namespace {

Tensor conv_pool_execute(const Tensor& input, const Tensor& weight, const Tensor* bias,
                         const ConvPoolShape& shape) {
    // 公共入口已经把操作数整理成连续布局；这里只兜住绕过入口的调用。
    const Tensor src = input.is_contiguous() ? input : input.clone();
    const Tensor w   = weight.is_contiguous() ? weight : weight.clone();

    Tensor result({shape.conv.batch, shape.conv.out_ch, shape.out_h, shape.out_w},
                  src.dtype(), Device::CUDA);

    const auto in_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!in_impl)
        throw std::runtime_error("cuda conv_pool: operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(in_impl->context());
    if (!ctx) throw std::runtime_error("cuda conv_pool: missing CUDAContext");

    const auto w_impl = std::dynamic_pointer_cast<CUDATensor>(w.get_impl());
    if (!w_impl)
        throw std::runtime_error("cuda conv_pool: weight does not live on the CUDA device");

    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!out_impl) throw std::runtime_error("cuda conv_pool: failed to allocate the result");

    const void* bias_ptr = nullptr;
    if (bias != nullptr) {
        const auto b_impl = std::dynamic_pointer_cast<CUDATensor>(bias->get_impl());
        if (!b_impl)
            throw std::runtime_error("cuda conv_pool: bias does not live on the CUDA device");
        bias_ptr = b_impl->data();
    }

    ctx->activate();
    tvv::cuda::check_cuda(
        ::tvv::cuda::launch_conv_pool(static_cast<cudaStream_t>(ctx->stream()),
                                      in_impl->data(), w_impl->data(), bias_ptr,
                                      out_impl->data(), src.dtype(), shape),
        "conv_pool launch");
    return result;
}

}  // namespace

Tensor Conv2dReluMaxPool2dImpl<Device::CUDA>::execute(const Tensor& input,
                                                      const Tensor& weight,
                                                      const Tensor* bias,
                                                      const ConvPoolShape& shape) {
    return conv_pool_execute(input, weight, bias, shape);
}

}  // namespace ops
