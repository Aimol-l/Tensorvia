#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_island.h"

// Host-side ufunc driver for SYCL: dtype promotion + NumPy broadcasting.
//
// The broadcast layout is derived by the device-independent helpers in
// :cpu_ops; strides and view offsets travel to the kernels through
// tvv::sycl::Bcast, so non-contiguous views are not materialized.
//
// This partition is the only host code that touches the seam. Everything below
// is an ordinary host function; the .sycl units live in a different compiler's
// entity universe, hence the int-status / void* contract.
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/sycl/sycl_context.hpp"
#include "interface/backend/sycl/sycl_tensor.hpp"

namespace ops::sycl_ufunc {

using ops::ufunc::broadcast_shapes;
using ops::ufunc::broadcast_strides;
using ops::ufunc::same_shape;

// Turn a seam status into the exception the other backends would have thrown.
inline void check(int status, const char* what) {
    if (status == ::tvv::sycl::kSyclOk) return;
    const char* reason = status == ::tvv::sycl::kSyclInvalid     ? "invalid argument"
                         : status == ::tvv::sycl::kSyclUnsupported ? "unsupported dtype"
                                                                  : "device failure";
    throw std::runtime_error(std::string("sycl ") + what + ": " + reason);
}

inline void* handle_of(const SYCLContext& ctx) {
    return ctx.handle();
}

inline void require_dims(size_t ndim, const char* what) {
    if (ndim > static_cast<size_t>(kMaxTensorDims)) {
        throw std::runtime_error(
            std::string("sycl ") + what + ": dimensions exceed the supported maximum of " +
            std::to_string(kMaxTensorDims));
    }
}

inline ::tvv::sycl::Bcast make_bcast_params(const Tensor& A, const Tensor& B,
                                            std::span<const int64_t> out_shape,
                                            const Tensor& out) {
    ::tvv::sycl::Bcast p{};
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
// temporary. Every SYCL launch is synchronous, so the temporary is fully
// consumed before it goes out of scope.
inline const Tensor& promote(const Tensor& t, DataType res, Tensor& storage) {
    if (t.dtype() == res) return t;
    storage = ops::Typecast(t, res);
    return storage;
}

inline std::shared_ptr<SYCLContext> context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx)
        throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
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
    if (dst == nullptr) res = Tensor(want, out_dtype, Device::SYCL);
    Tensor& out = dst != nullptr ? *dst : res;

    const auto A_impl = std::dynamic_pointer_cast<SYCLTensor>(A.get_impl());
    const auto B_impl = std::dynamic_pointer_cast<SYCLTensor>(B.get_impl());
    const auto O_impl = std::dynamic_pointer_cast<SYCLTensor>(out.get_impl());
    if (!A_impl || !B_impl || !O_impl)
        throw std::runtime_error("sycl binary op: operands must live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(A_impl->context());
    if (!ctx) throw std::runtime_error("sycl binary op: missing SYCLContext");

    ctx->activate();
    const ::tvv::sycl::Bcast p = make_bcast_params(A, B, want, out);
    check(::sycl_c_launch_binary(
              handle_of(*ctx), A_impl->data(), B_impl->data(), O_impl->data(),
              static_cast<int>(A.dtype()), op, &p),
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

    Tensor out(want, DataType::INT8, Device::SYCL);
    const auto A_impl = std::dynamic_pointer_cast<SYCLTensor>(A.get_impl());
    const auto B_impl = std::dynamic_pointer_cast<SYCLTensor>(B.get_impl());
    const auto O_impl = std::dynamic_pointer_cast<SYCLTensor>(out.get_impl());
    if (!A_impl || !B_impl || !O_impl)
        throw std::runtime_error("sycl comparison: operands must live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(A_impl->context());
    if (!ctx) throw std::runtime_error("sycl comparison: missing SYCLContext");

    double abs_tol = 0.0, rel_tol = 0.0;
    equality_tolerances(a.dtype(), b.dtype(), abs_tol, rel_tol);

    ctx->activate();
    const ::tvv::sycl::Bcast p = make_bcast_params(A, B, want, out);
    check(::sycl_c_launch_compare(
              handle_of(*ctx), A_impl->data(), B_impl->data(), O_impl->data(),
              static_cast<int>(A.dtype()), cmp, abs_tol, rel_tol, &p),
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

    Tensor out(A.shape(), DataType::INT8, Device::SYCL);
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(A.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<SYCLTensor>(out.get_impl());
    if (!impl || !out_impl)
        throw std::runtime_error("sycl scalar comparison: operands must live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error("sycl scalar comparison: missing SYCLContext");

    double abs_tol = 0.0, rel_tol = 0.0;
    scalar_tolerances(a.dtype(), abs_tol, rel_tol);

    ctx->activate();
    check(::sycl_c_launch_compare_scalar(
              handle_of(*ctx), A.data(), out.data(), static_cast<int64_t>(A.numel()),
              static_cast<int>(A.dtype()), cmp, static_cast<double>(value), abs_tol, rel_tol),
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
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": in-place update of a non-contiguous view; clone() it first");
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
    ctx->activate();
    check(::sycl_c_launch_scalar(
              handle_of(*ctx), a.data(), static_cast<int64_t>(a.numel()),
              static_cast<int>(a.dtype()), op, static_cast<double>(value)),
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
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": in-place update of a non-contiguous view; clone() it first");
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
    ctx->activate();
    check(::sycl_c_launch_unary(
              handle_of(*ctx), a.data(), a.data(), static_cast<int64_t>(a.numel()),
              static_cast<int>(a.dtype()), static_cast<int>(a.dtype()), op,
              static_cast<double>(p0), static_cast<double>(p1)),
          what);
    ctx->wait();
}

// Out-of-place unary: the result dtype is chosen by the operator, and the
// kernel reads the original dtype so integer inputs are still computed in
// double/float rather than being narrowed by an intermediate cast.
inline Tensor unary_out(const Tensor& a, DataType out_dtype, int op,
                        float p0, float p1, const char* what) {
    const Tensor& src = a.is_contiguous() ? a : a.clone();
    const auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(src.get_impl());
    if (!src_impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");

    Tensor out(src.shape(), out_dtype, Device::SYCL);
    ctx->activate();
    check(::sycl_c_launch_unary(
              handle_of(*ctx), src.data(), out.data(), static_cast<int64_t>(src.numel()),
              static_cast<int>(src.dtype()), static_cast<int>(out_dtype), op,
              static_cast<double>(p0), static_cast<double>(p1)),
          what);
    ctx->wait();
    return out;
}

}  // namespace ops::sycl_ufunc
