// CUDA 的宿主侧 ufunc 驱动：dtype 提升 + NumPy 广播。
//
// 广播布局由 ops_api/cpu_ops 中与设备无关的辅助函数推导；步长与视图 offset 通过
// tvv::cuda::Bcast 传给内核，因此非连续视图不会被物化。
//
// 本头文件只被 header-first 产物的 CUDA 翻译单元包含；模块 lane 看不到它。
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

// 无需转换时零拷贝，否则物化一个 Typecast 临时量。每次 CUDA 启动都是同步的，
// 所以临时量在离开作用域前已被完全消费。
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

// 通用广播启动。
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
    return res;
}

// 二元算术：dtype 提升 + NumPy 广播。
inline Tensor binary_arith(int op, const Tensor& a, const Tensor& b, Tensor* dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, res_type, dst);
}

// Equal/NotEqual 的容差按*原始*操作数的 dtype 选取，与 CPU 内核一致：提升到公共
// dtype 不得把精度档收紧。
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
        // 整型对在内核中精确比较；这两个值不会被使用。
        abs_tol = 0.0;   rel_tol = 0.0;
    }
}

// 标量比较把张量针对 FLOAT32 标量做提升，因此整型走浮点精度档（CPU 标量内核
// 同样如此）。
inline void scalar_tolerances(DataType a, double& abs_tol, double& rel_tol) {
    if (a == DataType::FLOAT64)        { abs_tol = 1e-9; rel_tol = 1e-12; }
    else if (a == DataType::FLOAT32)   { abs_tol = 1e-5; rel_tol = 1e-6; }
    else if (a == DataType::FLOAT16)   { abs_tol = 1e-3; rel_tol = 1e-3; }
    else if (a == DataType::BFLOAT16)  { abs_tol = 1e-2; rel_tol = 1e-2; }
    else                               { abs_tol = 1e-5; rel_tol = 1e-6; }
}

// 逻辑比较：提升到公共 dtype，比较，写入 BOOL。
inline Tensor compare_impl(int cmp, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);

    DimArray out_shape;
    if (!broadcast_shapes(A.shape(), B.shape(), out_shape))
        throw std::runtime_error("comparison: shapes are not broadcastable");
    const std::span<const int64_t> want(out_shape);

    Tensor out(want, DataType::BOOL, Device::CUDA);
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
    return out;
}

// 标量比较保留浮点张量的 dtype，弱类型（整型/BOOL）采用默认浮点 dtype，与 CPU 的
// 弱标量规则一致。
inline Tensor compare_scalar(int cmp, const Tensor& a, float value) {
    const DataType res_type = scalar_promote_type(a.dtype());
    Tensor storage;
    const Tensor& promoted = promote(a, res_type, storage);
    // 标量路径线性读取 A，所以跨步视图一次性物化，而不是给内核再教一套索引方式。
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
    // 弱类型（整型）张量在提升后的浮点精度档中比较。
    scalar_tolerances(is_integral_dtype(a.dtype()) ? res_type : a.dtype(),
                      abs_tol, rel_tol);

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_compare_scalar(stream_of(*ctx), A.data(), out.data(),
                                         static_cast<int64_t>(A.numel()), A.dtype(), cmp,
                                         static_cast<double>(value), abs_tol, rel_tol),
        "scalar comparison launch");
    return out;
}

// ---------------------------------------------------------------------------
// 标量与一元驱动。
//
// 内核只处理扁平连续 buffer，所以跨步视图会被物化（clone / to_type 都会产生连续
// 张量），而对视图的原地更新会被拒绝，以免破坏相邻元素。
// ---------------------------------------------------------------------------

// 其它后端把整型输入上的所有超越算子提升到 FLOAT32；Abs/Relu/Clamp 保持输入
// dtype。
inline DataType float_result(DataType t) {
    return is_integral_dtype(t) ? DataType::FLOAT32 : t;
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
    // Tensor::data() 是感知视图的指针（impl->data() 是 buffer 基址，对带存储
    // offset 的视图来说是错误地址）。
    tvv::cuda::check_cuda(
        tvv::cuda::launch_scalar(stream_of(*ctx), a.data(), static_cast<int64_t>(a.numel()),
                                 a.dtype(), op, value),
        what);
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
        tvv::cuda::launch_unary(stream_of(*ctx), a.data(), a.data(),
                                static_cast<int64_t>(a.numel()), a.dtype(), a.dtype(),
                                op, p0, p1),
        what);
}

// 非原地一元运算：结果 dtype 由算子决定，内核按原始 dtype 读取，因此整型输入仍在
// double/float 中计算，不会被中间 cast 收窄。
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
        tvv::cuda::launch_unary(stream_of(*ctx), src.data(), out.data(),
                                static_cast<int64_t>(src.numel()), src.dtype(), out_dtype,
                                op, p0, p1),
        what);
    return out;
}

}  // namespace ops::cuda_ufunc

#endif  // TVV_CUDA_UFUNC_DETAIL_HPP
