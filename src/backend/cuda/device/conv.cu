// 2-D 卷积的设备实现：一个线程算一个输出元素。
//
// 输出是连续 NCHW，所以线程的扁平下标直接解码成 (n, oc, oy, ox)，不需要任何
// 跨线程协作——卷积的每个输出都独立。权重与输入全程线性寻址；累加器取提升后的
// 计算类型，半精度输入不会在半精度里累积误差。
#include "common.cuh"

#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;
using tvv::cuda::contract::promoted_t;

template <class T>
__global__ void conv2d_kernel(const T* in, const T* weight, const T* bias, T* out,
                              ops::Conv2dShape p) {
    // 不用 Conv2dShape::numel()：它是宿主成员函数，设备代码里调不到。
    const int64_t total = p.batch * p.out_ch * p.out_h * p.out_w;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t ox   = idx % p.out_w;
    int64_t rest       = idx / p.out_w;
    const int64_t oy   = rest % p.out_h;
    rest /= p.out_h;
    const int64_t oc   = rest % p.out_ch;
    const int64_t n    = rest / p.out_ch;

    using Acc = promoted_t<T>;
    Acc acc = bias != nullptr ? tvv::cuda::to_compute<Acc>(bias[oc]) : Acc(0);

    const int64_t in_plane  = p.in_h * p.in_w;
    const int64_t out_plane = p.out_h * p.out_w;
    const int64_t taps      = p.in_ch * p.kh * p.kw;

    for (int64_t c = 0; c < p.in_ch; ++c) {
        const T* in_c = in + (n * p.in_ch + c) * in_plane;
        const T* w_c  = weight + oc * taps + c * p.kh * p.kw;
        for (int64_t i = 0; i < p.kh; ++i) {
            const int64_t y = oy * p.stride + i - p.padding;
            if (y < 0 || y >= p.in_h) continue;
            const T* in_row = in_c + y * p.in_w;
            const T* w_row  = w_c + i * p.kw;
            for (int64_t j = 0; j < p.kw; ++j) {
                const int64_t x = ox * p.stride + j - p.padding;
                if (x < 0 || x >= p.in_w) continue;
                acc += tvv::cuda::to_compute<Acc>(in_row[x]) *
                       tvv::cuda::to_compute<Acc>(w_row[j]);
            }
        }
    }

    out[(n * p.out_ch + oc) * out_plane + oy * p.out_w + ox] = tvv::cuda::store_as<T>(acc);
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_conv2d(cudaStream_t stream, const void* in, const void* weight,
                          const void* bias, void* out, DataType dtype,
                          const ops::Conv2dShape& p) {
    if (in == nullptr || weight == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        conv2d_kernel<T><<<blocks_for(p.numel()), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<const T*>(weight),
            static_cast<const T*>(bias), static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace ops {

Tensor Conv2dImpl<Device::CUDA>::execute(const Tensor& input, const Tensor& weight,
                                         const Tensor* bias, const Conv2dShape& shape) {
    using namespace ops::cuda_ufunc;

    // 公共入口已经把操作数整理成连续布局；这里只兜住绕过入口的调用。
    const Tensor src = input.is_contiguous() ? input : input.clone();
    const Tensor w   = weight.is_contiguous() ? weight : weight.clone();

    Tensor result({shape.batch, shape.out_ch, shape.out_h, shape.out_w}, src.dtype(),
                  Device::CUDA);

    const auto in_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!in_impl) throw std::runtime_error("cuda conv2d: operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(in_impl->context());
    if (!ctx) throw std::runtime_error("cuda conv2d: missing CUDAContext");

    const auto w_impl = std::dynamic_pointer_cast<CUDATensor>(w.get_impl());
    if (!w_impl) throw std::runtime_error("cuda conv2d: weight does not live on the CUDA device");

    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!out_impl) throw std::runtime_error("cuda conv2d: failed to allocate the result");

    const void* bias_ptr = nullptr;
    if (bias != nullptr) {
        const auto b_impl = std::dynamic_pointer_cast<CUDATensor>(bias->get_impl());
        if (!b_impl) throw std::runtime_error("cuda conv2d: bias does not live on the CUDA device");
        bias_ptr = b_impl->data();
    }

    ctx->activate();
    tvv::cuda::check_cuda(::tvv::cuda::launch_conv2d(stream_of(*ctx), in_impl->data(),
                                                    w_impl->data(), bias_ptr, out_impl->data(),
                                                    src.dtype(), shape),
                          "conv2d launch");
    return result;
}

}  // namespace ops
