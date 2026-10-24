// Element-wise arithmetic and activation kernels, and the host driver that
// defines AddImpl/SubImpl/MulImpl/DivImpl/AbsImpl/... for Device::CUDA.
//
// The host always promotes both operands (and the output) to a common dtype
// before calling in, so the entry points below can dispatch a single scalar
// type per launch. Mixed dtypes are rejected instead of silently narrowing.
#include "common.cuh"

#include <stdexcept>
#include <string>

#include <tensorvia/cuda_ops.h>
#include <tensorvia/ops_api.h>
#include <tensorvia/tensor.h>
#include <tensorvia/types.h>

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;
using tvv::cuda::scalar_from_double;
using tvv::cuda::store_as;
using tvv::cuda::to_compute;
using tvv::cuda::type_tag;

namespace contract = tvv::cuda::contract;

// Decode a flat output index against the broadcast shape, accumulating each
// operand's element offset with its own (already broadcast) stride array.
TVV_CUDA_HD int64_t bcast_offset(int64_t idx, const tvv::cuda::Bcast& p,
                                 const int64_t* strides) {
    int64_t off = 0;
    for (int d = p.ndim - 1; d >= 0; --d) {
        const int64_t dim = p.shape[d];
        if (dim <= 0) continue;
        off += (idx % dim) * strides[d];
        idx /= dim;
    }
    return off;
}

template <class C>
TVV_CUDA_HD C binary_value(int op, C x, C y) {
    switch (op) {
        case tvv::cuda::kBinAdd: return x + y;
        case tvv::cuda::kBinSub: return x - y;
        case tvv::cuda::kBinMul: return x * y;
        case tvv::cuda::kBinDiv: return contract::div_contract(x, y);
        default:                 return x;
    }
}

// op codes are uniform across the grid, so the branch is not per-lane divergence.
template <class T>
__global__ void binary_kernel(const T* a, const T* b, T* out, tvv::cuda::Bcast p, int op) {
    using C = contract::promoted_t<T>;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        const int64_t ia = p.off_a + bcast_offset(i, p, p.sa);
        const int64_t ib = p.off_b + bcast_offset(i, p, p.sb);
        const C x = to_compute<C>(a[ia]);
        const C y = to_compute<C>(b[ib]);
        out[p.off_o + i] = store_as<T>(binary_value(op, x, y));
    }
}

template <class T>
__global__ void scalar_kernel(T* buf, int64_t numel, int op, double scalar) {
    using C = contract::promoted_t<T>;
    const C s = scalar_from_double<C>(scalar);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        buf[i] = store_as<T>(binary_value(op, to_compute<C>(buf[i]), s));
    }
}

template <class C>
TVV_CUDA_HD C unary_value(int op, C x, C p0, C p1) {
    switch (op) {
        case tvv::cuda::kUnaryAbs:
            if constexpr (std::is_signed_v<C>) {
                return x < C(0) ? -x : x;
            } else {
                return x;
            }
        case tvv::cuda::kUnarySin:
            if constexpr (std::is_same_v<C, float>) return sinf(x);
            else if constexpr (std::is_same_v<C, double>) return ::sin(x);
            else return static_cast<C>(sin(static_cast<double>(x)));
        case tvv::cuda::kUnaryCos:
            if constexpr (std::is_same_v<C, float>) return cosf(x);
            else if constexpr (std::is_same_v<C, double>) return ::cos(x);
            else return static_cast<C>(cos(static_cast<double>(x)));
        case tvv::cuda::kUnaryTan:
            if constexpr (std::is_same_v<C, float>) return tanf(x);
            else if constexpr (std::is_same_v<C, double>) return ::tan(x);
            else return static_cast<C>(tan(static_cast<double>(x)));
        case tvv::cuda::kUnaryExp:
            if constexpr (std::is_same_v<C, float>) return expf(x);
            else if constexpr (std::is_same_v<C, double>) return ::exp(x);
            else return static_cast<C>(exp(static_cast<double>(x)));
        case tvv::cuda::kUnarySqrt:
            if constexpr (std::is_same_v<C, float>) return sqrtf(x);
            else if constexpr (std::is_same_v<C, double>) return ::sqrt(x);
            else return static_cast<C>(sqrt(static_cast<double>(x)));
        case tvv::cuda::kUnaryLog:
            // log base p0, matching the CPU backend: log(x) / log(base).
            if constexpr (std::is_same_v<C, float>) return logf(x) / logf(p0);
            else if constexpr (std::is_same_v<C, double>) return ::log(x) / log(p0);
            else return static_cast<C>(::log(static_cast<double>(x)) /
                                       ::log(static_cast<double>(p0)));
        case tvv::cuda::kUnaryPow:
            if constexpr (std::is_same_v<C, float>) return powf(x, p0);
            else if constexpr (std::is_same_v<C, double>) return ::pow(x, p0);
            else return static_cast<C>(::pow(static_cast<double>(x), static_cast<double>(p0)));
        case tvv::cuda::kUnaryRelu:
            return contract::relu_contract(x);
        case tvv::cuda::kUnarySilu: {
            // x * sigmoid(x), written so the tails saturate instead of overflowing.
            if constexpr (std::is_same_v<C, float>) return x / (1.0f + expf(-x));
            else return x / (1.0 + ::exp(-x));
        }
        case tvv::cuda::kUnaryTanh:
            if constexpr (std::is_same_v<C, float>) return tanhf(x);
            else if constexpr (std::is_same_v<C, double>) return ::tanh(x);
            else return static_cast<C>(tanh(static_cast<double>(x)));
        case tvv::cuda::kUnarySigmoid:
            if constexpr (std::is_same_v<C, float>) return 1.0f / (1.0f + expf(-x));
            else return 1.0 / (1.0 + ::exp(-x));
        case tvv::cuda::kUnaryClamp:
            return contract::clamp_contract(x, p0, p1);
        default:
            return x;
    }
}

template <class T, class R>
__global__ void unary_kernel(const T* in, R* out, int64_t numel, int op,
                             double p0, double p1) {
    using C = contract::unary_compute_t<T>;
    const C c0 = scalar_from_double<C>(p0);
    const C c1 = scalar_from_double<C>(p1);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        const C x = to_compute<C>(in[i]);
        out[i] = store_as<R>(unary_value(op, x, c0, c1));
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_binary(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype_a, DataType dtype_b,
                          DataType dtype_out, int op, const Bcast& p) {
    if (a == nullptr || b == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (dtype_a != dtype_b || dtype_a != dtype_out) return cudaErrorNotSupported;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype_a, [&](auto tag) {
        using T = typename decltype(tag)::type;
        binary_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out), p, op);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_scalar(cudaStream_t stream, void* buf, int64_t numel,
                          DataType dtype, int op, double scalar) {
    if (buf == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        scalar_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<T*>(buf), numel, op, scalar);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_unary(cudaStream_t stream, const void* in, void* out,
                         int64_t numel, DataType in_dtype, DataType out_dtype,
                         int op, double p0, double p1) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    // The only promotion the other backends apply is integer -> FLOAT32 and a
    // same-type in-place update, so those are the only two shapes instantiated.
    const bool same     = in_dtype == out_dtype;
    const bool to_float = out_dtype == DataType::FLOAT32;
    if (!same && !to_float) return cudaErrorNotSupported;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (same) {
            unary_kernel<T, T><<<blocks_for(numel), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), numel, op, p0, p1);
        } else {
            unary_kernel<T, float><<<blocks_for(numel), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<float*>(out), numel, op, p0, p1);
        }
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

namespace ops {

// ---------------------------------------------------------------------------
// Scalar arithmetic.
// ---------------------------------------------------------------------------
void AddImpl<Device::CUDA>::execute(Tensor& a, float b) {
    cuda_ufunc::scalar_inplace(a, tvv::cuda::kBinAdd, b, "Add");
}
void SubImpl<Device::CUDA>::execute(Tensor& a, float b) {
    cuda_ufunc::scalar_inplace(a, tvv::cuda::kBinSub, b, "Sub");
}
void MulImpl<Device::CUDA>::execute(Tensor& a, float b) {
    cuda_ufunc::scalar_inplace(a, tvv::cuda::kBinMul, b, "Mul");
}
void DivImpl<Device::CUDA>::execute(Tensor& a, float b) {
    cuda_ufunc::scalar_inplace(a, tvv::cuda::kBinDiv, b, "Div");
}

Tensor AddImpl<Device::CUDA>::execute(const Tensor& a, float b) {
    return cuda_ufunc::scalar_out(a, tvv::cuda::kBinAdd, b, "Add");
}
Tensor SubImpl<Device::CUDA>::execute(const Tensor& a, float b) {
    return cuda_ufunc::scalar_out(a, tvv::cuda::kBinSub, b, "Sub");
}
Tensor MulImpl<Device::CUDA>::execute(const Tensor& a, float b) {
    return cuda_ufunc::scalar_out(a, tvv::cuda::kBinMul, b, "Mul");
}
Tensor DivImpl<Device::CUDA>::execute(const Tensor& a, float b) {
    return cuda_ufunc::scalar_out(a, tvv::cuda::kBinDiv, b, "Div");
}

// ---------------------------------------------------------------------------
// Tensor-tensor arithmetic: dtype promotion + NumPy broadcasting.
// ---------------------------------------------------------------------------
Tensor AddImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::binary_arith(tvv::cuda::kBinAdd, a, b, nullptr);
}
Tensor SubImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::binary_arith(tvv::cuda::kBinSub, a, b, nullptr);
}
Tensor MulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::binary_arith(tvv::cuda::kBinMul, a, b, nullptr);
}
Tensor DivImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::binary_arith(tvv::cuda::kBinDiv, a, b, nullptr);
}

void AddImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    cuda_ufunc::binary_arith(tvv::cuda::kBinAdd, a, b, &dst);
}
void SubImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    cuda_ufunc::binary_arith(tvv::cuda::kBinSub, a, b, &dst);
}
void MulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    cuda_ufunc::binary_arith(tvv::cuda::kBinMul, a, b, &dst);
}
void DivImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    cuda_ufunc::binary_arith(tvv::cuda::kBinDiv, a, b, &dst);
}

// ---------------------------------------------------------------------------
// Unary element-wise. Integer inputs are computed in FLOAT32 for the
// transcendental operators; Abs/Clamp keep the input dtype.
// ---------------------------------------------------------------------------
namespace {

void unary_identity_inplace(Tensor& a, int op, float p0, float p1, const char* what) {
    cuda_ufunc::unary_inplace(a, op, p0, p1, what);
}
Tensor unary_identity_out(const Tensor& a, int op, float p0, float p1, const char* what) {
    return cuda_ufunc::unary_out(a, a.dtype(), op, p0, p1, what);
}
void unary_float_inplace(Tensor& a, int op, float p0, float p1, const char* what) {
    cuda_ufunc::unary_inplace(a, op, p0, p1, what);
}
Tensor unary_float_out(const Tensor& a, int op, float p0, float p1, const char* what) {
    return cuda_ufunc::unary_out(a, cuda_ufunc::float_result(a.dtype()), op, p0, p1, what);
}

}  // namespace

void AbsImpl<Device::CUDA>::execute(Tensor& a) {
    unary_identity_inplace(a, tvv::cuda::kUnaryAbs, 0.0f, 0.0f, "Abs");
}
Tensor AbsImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_identity_out(a, tvv::cuda::kUnaryAbs, 0.0f, 0.0f, "Abs");
}

void SinImpl<Device::CUDA>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::cuda::kUnarySin, 0.0f, 0.0f, "Sin");
}
Tensor SinImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::cuda::kUnarySin, 0.0f, 0.0f, "Sin");
}
void CosImpl<Device::CUDA>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::cuda::kUnaryCos, 0.0f, 0.0f, "Cos");
}
Tensor CosImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::cuda::kUnaryCos, 0.0f, 0.0f, "Cos");
}
void TanImpl<Device::CUDA>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::cuda::kUnaryTan, 0.0f, 0.0f, "Tan");
}
Tensor TanImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::cuda::kUnaryTan, 0.0f, 0.0f, "Tan");
}
void ExpImpl<Device::CUDA>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::cuda::kUnaryExp, 0.0f, 0.0f, "Exp");
}
Tensor ExpImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::cuda::kUnaryExp, 0.0f, 0.0f, "Exp");
}
void SqrtImpl<Device::CUDA>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::cuda::kUnarySqrt, 0.0f, 0.0f, "Sqrt");
}
Tensor SqrtImpl<Device::CUDA>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::cuda::kUnarySqrt, 0.0f, 0.0f, "Sqrt");
}

void PowImpl<Device::CUDA>::execute(Tensor& a, float val) {
    unary_float_inplace(a, tvv::cuda::kUnaryPow, val, 0.0f, "Pow");
}
Tensor PowImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return unary_float_out(a, tvv::cuda::kUnaryPow, val, 0.0f, "Pow");
}
void LogImpl<Device::CUDA>::execute(Tensor& a, float val) {
    unary_float_inplace(a, tvv::cuda::kUnaryLog, val, 0.0f, "Log");
}
Tensor LogImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return unary_float_out(a, tvv::cuda::kUnaryLog, val, 0.0f, "Log");
}

void ClampImpl<Device::CUDA>::execute(Tensor& a, float min, float max) {
    unary_identity_inplace(a, tvv::cuda::kUnaryClamp, min, max, "Clamp");
}
Tensor ClampImpl<Device::CUDA>::execute(const Tensor& a, float min, float max) {
    return unary_identity_out(a, tvv::cuda::kUnaryClamp, min, max, "Clamp");
}

}  // namespace ops
