// Host-side ufunc driver for CUDA: dtype promotion + NumPy broadcasting.
//
// The broadcast layout is derived by the device-independent helpers in
// ops_api/cpu_ops; strides and view offsets travel to the kernels through
// tvv::cuda::Bcast, so non-contiguous views are not materialized.
//
// This header is only included by the CUDA translation units of the
// header-first artifact; the module lane never sees it.
#ifndef TVV_CUDA_UFUNC_DETAIL_HPP
#define TVV_CUDA_UFUNC_DETAIL_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace ops::cuda_ufunc {

using ops::ufunc::broadcast_shapes;
using ops::ufunc::broadcast_strides;
using ops::ufunc::same_shape;

inline cudaStream_t stream_of(const CUDAContext& ctx) {
    return static_cast<cudaStream_t>(ctx.stream());
}

inline void require_dims(size_t ndim, const char* what) {
    if (ndim > static_cast<size_t>(kMaxTensorDims)) {
        throw std::runtime_error(
            std::string("cuda ") + what + ": dimensions exceed the supported maximum of " +
            std::to_string(kMaxTensorDims));
    }
}

inline tvv::cuda::Bcast make_bcast_params(const Tensor& A, const Tensor& B,
                                          std::span<const int64_t> out_shape,
                                          const Tensor& out) {
    tvv::cuda::Bcast p{};
    const size_t ndim = out_shape.size();
    require_dims(ndim, "binary op");
    p.ndim = static_cast<int32_t>(ndim);

    int64_t total = 1;
    for (int64_t d : out_shape) total *= d;
    p.numel = total;

    const auto sa = broadcast_strides(A.shape(), A.strides(), ndim);
    const auto sb = broadcast_strides(B.shape(), B.strides(), ndim);
    for (size_t d = 0; d < ndim; ++d) {
        p.shape[d] = out_shape[d];
        p.sa[d]    = sa[d];
        p.sb[d]    = sb[d];
    }
    p.off_a = A.offset();
    p.off_b = B.offset();
    p.off_o = out.offset();
    return p;
}

// Zero-copy when no conversion is needed, otherwise a materialized Typecast
// temporary. Every CUDA launch is synchronous, so the temporary is fully
// consumed before it goes out of scope.
inline const Tensor& promote(const Tensor& t, DataType res, Tensor& storage) {
    if (t.dtype() == res) return t;
    storage = ops::Typecast(t, res);
    return storage;
}

inline std::shared_ptr<CUDAContext> context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx)
        throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    return ctx;
}

// Generic broadcast launch.
inline Tensor bcast_binary(int op, const Tensor& A, const Tensor& B, DataType out_dtype,
                           Tensor* dst) {
    DimArray out_shape;
    if (!broadcast_shapes(A.shape(), B.shape(), out_shape))
        throw std::runtime_error("binary op: shapes are not broadcastable");
    const std::span<const int64_t> want(out_shape);

    if (dst != nullptr) {
        if (!dst->is_contiguous())
            throw std::runtime_error(
                "dst: non-contiguous view is not supported here; call to_contiguous() first");
        if (dst->dtype() != out_dtype)
            throw std::runtime_error("dst dtype must equal the promoted result dtype");
        if (!same_shape(dst->shape(), want))
            throw std::runtime_error("dst shape must equal the broadcast result shape");
    }
    Tensor res;
    if (dst == nullptr) res = Tensor(want, out_dtype, Device::CUDA);
    Tensor& out = dst != nullptr ? *dst : res;

    const auto A_impl = std::dynamic_pointer_cast<CUDATensor>(A.get_impl());
    const auto B_impl = std::dynamic_pointer_cast<CUDATensor>(B.get_impl());
    const auto O_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!A_impl || !B_impl || !O_impl)
        throw std::runtime_error("cuda binary op: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(A_impl->context());
    if (!ctx) throw std::runtime_error("cuda binary op: missing CUDAContext");

    ctx->activate();
    const tvv::cuda::Bcast p = make_bcast_params(A, B, want, out);
    tvv::cuda::check_cuda(
        tvv::cuda::launch_binary(stream_of(*ctx), A_impl->data(), B_impl->data(),
                                 O_impl->data(), A.dtype(), B.dtype(), out_dtype, op, p),
        "binary launch");
    ctx->wait();
    return res;
}

// Binary arithmetic: dtype promotion + NumPy broadcasting.
inline Tensor binary_arith(int op, const Tensor& a, const Tensor& b, Tensor* dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, res_type, dst);
}

// Equal/NotEqual tolerances are chosen from the *original* operand dtypes, just
// like the CPU kernel: promotion to a common dtype must not tighten the tier.
inline void equality_tolerances(DataType a, DataType b, double& abs_tol, double& rel_tol) {
    if (a == DataType::FLOAT64 || b == DataType::FLOAT64) {
        abs_tol = 1e-9;  rel_tol = 1e-12;
    } else if (a == DataType::FLOAT32 || b == DataType::FLOAT32) {
        abs_tol = 1e-5;  rel_tol = 1e-6;
    } else if (a == DataType::FLOAT16 || b == DataType::FLOAT16) {
        abs_tol = 1e-3;  rel_tol = 1e-3;
    } else if (a == DataType::BFLOAT16 || b == DataType::BFLOAT16) {
        abs_tol = 1e-2;  rel_tol = 1e-2;
    } else {
        // Integral pairs compare exactly in the kernel; the values are unused.
        abs_tol = 0.0;   rel_tol = 0.0;
    }
}

// Scalar comparison promotes the tensor against a FLOAT32 scalar, so integers
// take the float tier (the CPU scalar kernel does the same).
inline void scalar_tolerances(DataType a, double& abs_tol, double& rel_tol) {
    if (a == DataType::FLOAT64)        { abs_tol = 1e-9; rel_tol = 1e-12; }
    else if (a == DataType::FLOAT32)   { abs_tol = 1e-5; rel_tol = 1e-6; }
    else if (a == DataType::FLOAT16)   { abs_tol = 1e-3; rel_tol = 1e-3; }
    else if (a == DataType::BFLOAT16)  { abs_tol = 1e-2; rel_tol = 1e-2; }
    else                               { abs_tol = 1e-5; rel_tol = 1e-6; }
}

// Logical comparison: promote to a common dtype, compare, write INT8.
inline Tensor compare_impl(int cmp, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);

    DimArray out_shape;
    if (!broadcast_shapes(A.shape(), B.shape(), out_shape))
        throw std::runtime_error("comparison: shapes are not broadcastable");
    const std::span<const int64_t> want(out_shape);

    Tensor out(want, DataType::INT8, Device::CUDA);
    const auto A_impl = std::dynamic_pointer_cast<CUDATensor>(A.get_impl());
    const auto B_impl = std::dynamic_pointer_cast<CUDATensor>(B.get_impl());
    const auto O_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!A_impl || !B_impl || !O_impl)
        throw std::runtime_error("cuda comparison: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(A_impl->context());
    if (!ctx) throw std::runtime_error("cuda comparison: missing CUDAContext");

    double abs_tol = 0.0, rel_tol = 0.0;
    equality_tolerances(a.dtype(), b.dtype(), abs_tol, rel_tol);

    ctx->activate();
    const tvv::cuda::Bcast p = make_bcast_params(A, B, want, out);
    tvv::cuda::check_cuda(
        tvv::cuda::launch_compare(stream_of(*ctx), A.data(), B.data(), out.data(),
                                  A.dtype(), B.dtype(), cmp, abs_tol, rel_tol, p),
        "comparison launch");
    ctx->wait();
    return out;
}

// Scalar comparison over a contiguous input (the scalar broadcasts).
inline Tensor compare_scalar(int cmp, const Tensor& a, float value) {
    const DataType res_type = compute_type(a.dtype(), DataType::FLOAT32);
    Tensor storage;
    const Tensor& promoted = promote(a, res_type, storage);
    // The scalar path reads A linearly, so a strided view is materialized once
    // rather than teaching the kernel a second indexing scheme.
    Tensor contiguous_storage;
    const Tensor& A = [&]() -> const Tensor& {
        if (promoted.is_contiguous()) return promoted;
        contiguous_storage = promoted.contiguous();
        return contiguous_storage;
    }();

    Tensor out(A.shape(), DataType::INT8, Device::CUDA);
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(A.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!impl || !out_impl)
        throw std::runtime_error("cuda scalar comparison: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error("cuda scalar comparison: missing CUDAContext");

    double abs_tol = 0.0, rel_tol = 0.0;
    scalar_tolerances(a.dtype(), abs_tol, rel_tol);

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_compare_scalar(stream_of(*ctx), A.data(), out.data(),
                                         static_cast<int64_t>(A.numel()), A.dtype(), cmp,
                                         static_cast<double>(value), abs_tol, rel_tol),
        "scalar comparison launch");
    ctx->wait();
    return out;
}

// ---------------------------------------------------------------------------
// Scalar and unary drivers.
//
// The kernels work on flat, contiguous buffers only, so a strided view is
// materialized (clone / to_type both produce a contiguous tensor) and an
// in-place update of a view is refused instead of corrupting its neighbours.
// ---------------------------------------------------------------------------

// The other backends promote every transcendental operator on integer inputs
// to FLOAT32; Abs/Relu/Clamp keep the input dtype.
inline DataType float_result(DataType t) {
    return t <= DataType::INT64 ? DataType::FLOAT32 : t;
}

inline void scalar_inplace(Tensor& a, int op, float value, const char* what) {
    if (!a.is_contiguous())
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": in-place update of a non-contiguous view; clone() it first");
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_scalar(stream_of(*ctx), impl->data(), static_cast<int64_t>(a.numel()),
                                 a.dtype(), op, value),
        what);
    ctx->wait();
}

inline Tensor scalar_out(const Tensor& a, int op, float value, const char* what) {
    Tensor result = a.clone();
    scalar_inplace(result, op, value, what);
    return result;
}

inline void unary_inplace(Tensor& a, int op, float p0, float p1, const char* what) {
    if (!a.is_contiguous())
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": in-place update of a non-contiguous view; clone() it first");
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_unary(stream_of(*ctx), impl->data(), impl->data(),
                                static_cast<int64_t>(a.numel()), a.dtype(), a.dtype(),
                                op, p0, p1),
        what);
    ctx->wait();
}

// Out-of-place unary: the result dtype is chosen by the operator, and the
// kernel reads the original dtype so integer inputs are still computed in
// double/float rather than being narrowed by an intermediate cast.
inline Tensor unary_out(const Tensor& a, DataType out_dtype, int op,
                        float p0, float p1, const char* what) {
    const Tensor& src = a.is_contiguous() ? a : a.clone();
    const auto src_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!src_impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");

    Tensor out(src.shape(), out_dtype, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate the result");
    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_unary(stream_of(*ctx), src_impl->data(), out_impl->data(),
                                static_cast<int64_t>(src.numel()), src.dtype(), out_dtype,
                                op, p0, p1),
        what);
    ctx->wait();
    return out;
}

}  // namespace ops::cuda_ufunc

#endif  // TVV_CUDA_UFUNC_DETAIL_HPP
