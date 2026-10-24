// Activation kernels that need more than the element-wise unary path.
// Relu/Silu/Tanh/Sigmoid live in arithmetic.cu; Softmax reduces along one axis
// and is implemented here, together with the host driver for all of them.
#include "common.cuh"

#include <stdexcept>

#include <tensorvia/cuda_ops.h>
#include <tensorvia/ops_api.h>
#include <tensorvia/tensor.h>
#include <tensorvia/types.h>

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace tvv::cuda {
namespace {

template <class C>
TVV_CUDA_HD C softmax_exp(C v) {
    if constexpr (std::is_same_v<C, double>) return ::exp(v);
    else return expf(v);
}

template <class T, class R>
__global__ void softmax_kernel(const T* in, R* out, tvv::cuda::ReduceAxis p) {
    using C = std::conditional_t<std::is_same_v<R, double>, double, float>;
    const int64_t total = p.outer * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t outer_idx = idx / p.inner;
        const int64_t inner_idx = idx % p.inner;
        const int64_t base = outer_idx * p.axis * p.inner + inner_idx;

        C max_val = -inf_value<C>();
        for (int64_t i = 0; i < p.axis; ++i) {
            const C v = to_compute<C>(in[base + i * p.inner]);
            max_val = v > max_val ? v : max_val;
        }
        C sum = C(0);
        for (int64_t i = 0; i < p.axis; ++i) {
            const C v = to_compute<C>(in[base + i * p.inner]);
            sum += softmax_exp<C>(v - max_val);
        }
        for (int64_t i = 0; i < p.axis; ++i) {
            const C v = to_compute<C>(in[base + i * p.inner]);
            const C e = softmax_exp<C>(v - max_val);
            out[base + i * p.inner] = store_as<R>(e / sum);
        }
    }
}

}  // namespace

cudaError_t launch_softmax(cudaStream_t stream, const void* in, void* out,
                           DataType in_dtype, DataType out_dtype,
                           const ReduceAxis& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    const int64_t total = p.outer * p.inner;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using R = typename decltype(dst_tag)::type;
            softmax_kernel<T, R><<<blocks_for(total), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<R*>(out), p);
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

// Softmax reads and writes one reduction axis at a time; the kernel works on a
// flat row-major buffer, so a strided operand is materialized first.
Tensor softmax_impl(const Tensor& a, int axis) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error("cuda Softmax: axis out of range");

    const Tensor src = a.is_contiguous() ? a : a.clone();
    tvv::cuda::ReduceAxis p{};
    p.axis  = a.shape(axis);
    p.inner = 1;
    for (int i = axis + 1; i < dim; ++i) p.inner *= a.shape(i);
    p.outer = 1;
    for (int i = 0; i < axis; ++i) p.outer *= a.shape(i);

    // Integers have no meaningful probability output; the other backends
    // promote them to FLOAT32, everything else keeps its storage type.
    const DataType out_type =
        src.dtype() <= DataType::INT64 ? DataType::FLOAT32 : src.dtype();
    Tensor out(src.shape(), out_type, Device::CUDA);

    auto ctx = cuda_ufunc::context_of(src, "Softmax");
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error("cuda Softmax: failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_softmax(cuda_ufunc::stream_of(*ctx), src.data(), out.data(),
                                  src.dtype(), out_type, p),
        "Softmax launch");
    ctx->wait();
    return out;
}

}  // namespace

void ReluImpl<Device::CUDA>::execute(Tensor& a) {
    cuda_ufunc::unary_inplace(a, tvv::cuda::kUnaryRelu, 0.0f, 0.0f, "Relu");
}
Tensor ReluImpl<Device::CUDA>::execute(const Tensor& a) {
    return cuda_ufunc::unary_out(a, a.dtype(), tvv::cuda::kUnaryRelu, 0.0f, 0.0f, "Relu");
}

void SiluImpl<Device::CUDA>::execute(Tensor& a) {
    cuda_ufunc::unary_inplace(a, tvv::cuda::kUnarySilu, 0.0f, 0.0f, "Silu");
}
Tensor SiluImpl<Device::CUDA>::execute(const Tensor& a) {
    return cuda_ufunc::unary_out(a, cuda_ufunc::float_result(a.dtype()),
                                 tvv::cuda::kUnarySilu, 0.0f, 0.0f, "Silu");
}

void TanhImpl<Device::CUDA>::execute(Tensor& a) {
    cuda_ufunc::unary_inplace(a, tvv::cuda::kUnaryTanh, 0.0f, 0.0f, "Tanh");
}
Tensor TanhImpl<Device::CUDA>::execute(const Tensor& a) {
    return cuda_ufunc::unary_out(a, cuda_ufunc::float_result(a.dtype()),
                                 tvv::cuda::kUnaryTanh, 0.0f, 0.0f, "Tanh");
}

void SigmoidImpl<Device::CUDA>::execute(Tensor& a) {
    cuda_ufunc::unary_inplace(a, tvv::cuda::kUnarySigmoid, 0.0f, 0.0f, "Sigmoid");
}
Tensor SigmoidImpl<Device::CUDA>::execute(const Tensor& a) {
    return cuda_ufunc::unary_out(a, cuda_ufunc::float_result(a.dtype()),
                                 tvv::cuda::kUnarySigmoid, 0.0f, 0.0f, "Sigmoid");
}

Tensor SoftmaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return softmax_impl(a, axis);
}

}  // namespace ops
