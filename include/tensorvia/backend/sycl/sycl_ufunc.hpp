#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_island.h"

// SYCL 的宿主侧 ufunc 驱动：dtype 提升 + NumPy 广播。
//
// 广播布局由 :cpu_ops 里与设备无关的助手推导；strides 与视图 offset 通过
// tvv::sycl::Bcast 传给内核，所以非连续视图不必物化。
//
// 这个分区是唯一接触 seam 的宿主代码。下面全是普通宿主函数；.sycl 单元活在另一个
// 编译器的实体宇宙里，因此约定用 int 状态码 / void* 接口。
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/tensor_iterator.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"

namespace ops::sycl_ufunc {

using ops::iter::broadcast_shapes;
using ops::iter::broadcast_strides;
using ops::iter::same_shape;
using ops::iter::Layout;
using ops::iter::promote;

// 把 seam 状态码变成其它后端本来会抛的异常。
inline void check(int status, const char* what) {
    if (status == ::tvv::sycl::kSyclOk) return;
    const char* reason = status == ::tvv::sycl::kSyclInvalid     ? "invalid argument"
                         : status == ::tvv::sycl::kSyclUnsupported ? "unsupported dtype"
                                                                  : "device failure";
    // 设备运行时抛了什么就带出来：接缝只有状态码，"device failure" 对排查毫无
    // 用处，"No kernel named ... was found" 才能定位到设备镜像/选择器的问题。
    const char* detail = ::sycl_c_last_error();
    if (detail == nullptr || detail[0] == '\0')
        throw std::runtime_error(std::string("sycl ") + what + ": " + reason);
    throw std::runtime_error(std::string("sycl ") + what + ": " + reason + " (" + detail + ")");
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

// 只做"布局 → 参数结构体"的翻译；偏移量随参数走，内核从 impl 基址寻址。
// linearizable 布局把 ndim 置 0：内核据此走连续快路径（两侧 stride 恒为 1），
// 跳过每元素的 stride 解码；shape/strides 留原值，快路径不读它们。
inline ::tvv::sycl::Bcast make_bcast_params(const Layout& lay, const Tensor& out) {
    ::tvv::sycl::Bcast p{};
    require_dims(static_cast<size_t>(lay.ndim), "binary op");
    p.ndim  = iter::linearizable(lay) ? 0 : lay.ndim;
    p.numel = lay.numel;
    for (int32_t d = 0; d < lay.ndim; ++d) {
        p.shape[d] = lay.shape[d];
        p.sa[d]    = lay.operands[0].strides[d];
        p.sb[d]    = lay.operands[1].strides[d];
    }
    p.off_a = lay.operands[0].offset;
    p.off_b = lay.operands[1].offset;
    p.off_o = out.offset();
    return p;
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

// 后端策略：把统一迭代层的 Layout 翻译成 seam 参数并发射。
struct SyclElementwise {
    using Impl = SYCLTensor;
    using Ctx  = SYCLContext;

    static constexpr Device           device = Device::SYCL;
    static constexpr std::string_view name   = "sycl";

    static void activate(Ctx& ctx) { ctx.activate(); }

    template <class Op>
    static void launch(Ctx& ctx, const Tensor& A, const Tensor& B, const Tensor& out,
                       const std::shared_ptr<Impl>& ai, const std::shared_ptr<Impl>& bi,
                       const std::shared_ptr<Impl>& oi, DataType, Op op, const Layout& lay) {
        const ::tvv::sycl::Bcast p = make_bcast_params(lay, out);
        check(::sycl_c_launch_binary(handle_of(ctx), ai->data(), bi->data(), oi->data(),
                                     static_cast<int>(A.dtype()), op, &p),
              "binary launch");
    }
};

// 通用的广播 launch。
inline Tensor bcast_binary(int op, const Tensor& A, const Tensor& B, DataType out_dtype,
                           Tensor* dst) {
    return ops::iter::launch_binary<SyclElementwise>(op, A, B, out_dtype, dst);
}

// 二元算术：dtype 提升 + NumPy 广播。
inline Tensor binary_arith(int op, const Tensor& a, const Tensor& b, Tensor* dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, res_type, dst);
}

// Equal/NotEqual 的容差取自*原始*操作数 dtype，与 CPU 内核一致：提升到公共 dtype
// 不能让容差档位变紧。
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
        // 整型对在内核里精确比较；这两个值用不到。
        abs_tol = 0.0;   rel_tol = 0.0;
    }
}

// 标量比较把张量与 FLOAT32 标量做提升，因此整型走浮点容差档（CPU 标量内核同理）。
inline void scalar_tolerances(DataType a, double& abs_tol, double& rel_tol) {
    if (a == DataType::FLOAT64)        { abs_tol = 1e-9; rel_tol = 1e-12; }
    else if (a == DataType::FLOAT32)   { abs_tol = 1e-5; rel_tol = 1e-6; }
    else if (a == DataType::FLOAT16)   { abs_tol = 1e-3; rel_tol = 1e-3; }
    else if (a == DataType::BFLOAT16)  { abs_tol = 1e-2; rel_tol = 1e-2; }
    else                               { abs_tol = 1e-5; rel_tol = 1e-6; }
}

// 逻辑比较：提升到公共 dtype、比较、写 BOOL。
inline Tensor compare_impl(int cmp, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);

    const Layout lay = ops::iter::build_binary(A, B, "comparison");
    const std::span<const int64_t> want(lay.shape);

    Tensor out(want, DataType::BOOL, Device::SYCL);
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
    const ::tvv::sycl::Bcast p = make_bcast_params(lay, out);
    check(::sycl_c_launch_compare(
              handle_of(*ctx), A_impl->data(), B_impl->data(), O_impl->data(),
              static_cast<int>(A.dtype()), cmp, abs_tol, rel_tol, &p),
          "comparison launch");
    return out;
}

// 标量比较保持浮点张量的 dtype，弱类型（整型/BOOL）采用默认浮点 dtype，
// 与 CPU 的弱标量规则一致。
inline Tensor compare_scalar(int cmp, const Tensor& a, float value) {
    const DataType res_type = scalar_promote_type(a.dtype());
    Tensor storage;
    const Tensor& promoted = promote(a, res_type, storage);
    // 标量路径线性读取 A，所以跨步视图物化一次，而不是教内核第二套寻址方案。
    Tensor contiguous_storage;
    const Tensor& A = [&]() -> const Tensor& {
        if (promoted.is_contiguous()) return promoted;
        contiguous_storage = promoted.contiguous();
        return contiguous_storage;
    }();

    Tensor out(A.shape(), DataType::BOOL, Device::SYCL);
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(A.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<SYCLTensor>(out.get_impl());
    if (!impl || !out_impl)
        throw std::runtime_error("sycl scalar comparison: operands must live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error("sycl scalar comparison: missing SYCLContext");

    double abs_tol = 0.0, rel_tol = 0.0;
    // 弱类型（整型）张量按提升后的浮点档比较。
    scalar_tolerances(is_integral_dtype(a.dtype()) ? res_type : a.dtype(),
                      abs_tol, rel_tol);

    ctx->activate();
    check(::sycl_c_launch_compare_scalar(
              handle_of(*ctx), A.data(), out.data(), static_cast<int64_t>(A.numel()),
              static_cast<int>(A.dtype()), cmp, static_cast<double>(value), abs_tol, rel_tol),
          "scalar comparison launch");
    return out;
}

// ---------------------------------------------------------------------------
// 标量与一元驱动。
//
// 内核只处理扁平连续 buffer，所以跨步视图会被物化（clone / to_type 都产生连续
// 张量），而在视图上原地更新会被拒绝，而不是写坏相邻元素。
// ---------------------------------------------------------------------------

// 其它后端把所有超越算子作用在整型输入时提升到 FLOAT32；Abs/Relu/Clamp 保持输入
// dtype。
inline DataType float_result(DataType t) {
    return is_integral_dtype(t) ? DataType::FLOAT32 : t;
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
}

// 非原地一元：结果 dtype 由算子决定，内核按原始 dtype 读入，整型输入仍在
// double/float 里计算，不会被中间 cast 截断。
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
    return out;
}

}  // namespace ops::sycl_ufunc
