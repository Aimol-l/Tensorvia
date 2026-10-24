#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"

namespace ops {

template <Device D> struct AddImpl;
template <Device D> struct SubImpl;
template <Device D> struct MulImpl;
template <Device D> struct DivImpl;
template <Device D> struct SinImpl;
template <Device D> struct CosImpl;
template <Device D> struct TanImpl;
template <Device D> struct PowImpl;
template <Device D> struct LogImpl;
template <Device D> struct ExpImpl;
template <Device D> struct SqrtImpl;
template <Device D> struct AbsImpl;
template <Device D> struct ClampImpl;

template <Device D> struct ZerosImpl;
template <Device D> struct OnesImpl;
template <Device D> struct FillImpl;
template <Device D> struct RandomImpl;
template <Device D> struct ArangeImpl;
template <Device D> struct LinspaceImpl;
template <Device D> struct EyeImpl;
template <Device D> struct RandnImpl;
template <Device D> struct WhereImpl;
template <Device D> struct IndexSelectImpl;
template <Device D> struct GatherImpl;

template <Device D> struct SumImpl;
template <Device D> struct MeanImpl;
template <Device D> struct MaxImpl;
template <Device D> struct MinImpl;
template <Device D> struct ArgMaxImpl;
template <Device D> struct ArgMinImpl;
template <Device D> struct AllImpl;
template <Device D> struct AnyImpl;
template <Device D> struct ProdImpl;
template <Device D> struct VarImpl;
template <Device D> struct StdImpl;
template <Device D> struct NormImpl;
template <Device D> struct CumsumImpl;
template <Device D> struct LogSoftmaxImpl;
template <Device D> struct ErfImpl;
template <Device D> struct MseLossImpl;
template <Device D> struct NllLossImpl;

template <Device D> struct EqualImpl;
template <Device D> struct NotEqualImpl;
template <Device D> struct GreaterImpl;
template <Device D> struct LessImpl;
template <Device D> struct GreaterEqualImpl;
template <Device D> struct LessEqualImpl;
template <Device D> struct NonZeroImpl;

template <Device D> struct ReluImpl;
template <Device D> struct SiluImpl;
template <Device D> struct TanhImpl;
template <Device D> struct SigmoidImpl;
template <Device D> struct SoftmaxImpl;
template <Device D> struct LayerNormImpl;
template <Device D> struct RmsNormImpl;

template <> struct AddImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct SubImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct MulImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct DivImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct AbsImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SinImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct CosImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct ExpImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SqrtImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct LogImpl<Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct PowImpl<Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct ClampImpl<Device::CPU> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

template <> struct ZerosImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct OnesImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct FillImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
    static void execute(Tensor& a, float value);
};
template <> struct RandomImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max);
};
template <> struct RandnImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float mean, float std);
};
template <> struct ArangeImpl<Device::CPU> {
    static Tensor execute(float start, float stop, float step, DataType dtype);
};
template <> struct LinspaceImpl<Device::CPU> {
    static Tensor execute(float start, float stop, int64_t steps, DataType dtype);
};
template <> struct EyeImpl<Device::CPU> {
    static Tensor execute(int64_t n, int64_t m, DataType dtype);
};
template <> struct WhereImpl<Device::CPU> {
    static Tensor execute(const Tensor& cond, const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& cond, const Tensor& a, float b);
    static Tensor execute(const Tensor& cond, float a, const Tensor& b);
};
template <> struct IndexSelectImpl<Device::CPU> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};
template <> struct GatherImpl<Device::CPU> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};

template <> struct ReluImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SiluImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanhImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SigmoidImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SoftmaxImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct LogSoftmaxImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ErfImpl<Device::CPU> {
    static Tensor execute(const Tensor& a);
};
template <> struct LayerNormImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps);
};
template <> struct RmsNormImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& weight, int axis, float eps);
};

template <> struct SumImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MinImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MaxImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MeanImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ProdImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct VarImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct StdImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct NormImpl<Device::CPU> {
    static float execute(const Tensor& a, float p);
    static Tensor execute(const Tensor& a, float p, int axis);
};
template <> struct CumsumImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MseLossImpl<Device::CPU> {
    static float execute(const Tensor& input, const Tensor& target);
};
template <> struct NllLossImpl<Device::CPU> {
    static float execute(const Tensor& input, const Tensor& target);
};
template <> struct ArgMaxImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMinImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct AllImpl<Device::CPU> {
    static bool execute(const Tensor& a, float value);
};
template <> struct AnyImpl<Device::CPU> {
    static bool execute(const Tensor& a, float value);
};

template <Device D> struct TypecastImpl;
template <> struct TypecastImpl<Device::CPU> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

template <Device D> struct ConcatImpl;
template <> struct ConcatImpl<Device::CPU> {
    static Tensor execute(const std::vector<Tensor>& tensors, int dim);
};

template <Device D> struct TransposeImpl;
template <> struct TransposeImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);
};

template <Device D> struct SliceImpl;
template <> struct SliceImpl<Device::CPU> {
    static Tensor execute(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);
};

template <Device D> struct MatMulImpl;
template <> struct MatMulImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};

template <> struct EqualImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessEqualImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterEqualImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NotEqualImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NonZeroImpl<Device::CPU> {
    static size_t execute(const Tensor& a);
};

template <Device D> struct PrintlnImpl;
template <> struct PrintlnImpl<Device::CPU> {
    static void execute(const Tensor& a);
};

template <Device D> struct ExportCsvImpl;
template <> struct ExportCsvImpl<Device::CPU> {
    static void execute(const Tensor& a, const std::string& path);
};

template <Device D> struct TempImpl;
template <> struct TempImpl<Device::CPU> {
    static void execute(Tensor& a);
};

// ================================================================
// element-wise ufunc 框架：
// 类型提升 / 非连续视图物化 / 输出分配 / dtype 分发统一在这里处理，
// 各算子只需提供类型特化 kernel（Op 结构体），不再复制粘贴调度代码。
// 这是 NumPy ufunc 分层的 CPU 版：框架 = 迭代器 + 类型解析，Op = inner loop。
namespace ufunc {

// 非连续视图先物化成连续拷贝（kernel 只支持线性遍历；
// strided 内核接入前的正确性守卫）。storage 保活物化结果。
inline const Tensor& linearize(const Tensor& t, Tensor& storage) {
    if (t.is_contiguous()) return t;
    storage = t.contiguous();
    return storage;
}

// 写入型操作（in-place / dst 输出）在非连续视图上线性读写会破坏
// 视图外的共享内存 —— 明确拒绝。
inline void require_contiguous(const Tensor& t, const char* op) {
    if (!t.is_contiguous())
        throw std::runtime_error(std::format(
            "{}: non-contiguous view is not supported here; call to_contiguous() first", op));
}

// span 没有 operator==（C++20 设计如此），统一走这个助手
inline bool same_shape(std::span<const int64_t> a, std::span<const int64_t> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// NumPy 广播规则：shape 右对齐逐维比较，相等或一方为 1 可广播
inline bool broadcast_shapes(std::span<const int64_t> sa, std::span<const int64_t> sb,
                             DimArray& out) {
    const size_t na = sa.size(), nb = sb.size();
    const size_t n = std::max(na, nb);
    out.clear();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = 1;
    for (size_t i = 0; i < n; ++i) {
        const int64_t da = (i + na < n) ? 1 : sa[i - (n - na)];
        const int64_t db = (i + nb < n) ? 1 : sb[i - (n - nb)];
        if (da == db)      out[i] = da;
        else if (da == 1)  out[i] = db;
        else if (db == 1)  out[i] = da;
        else return false;
    }
    return true;
}

// 对齐到输出 ndim 后的有效 strides：缺失维度或大小为 1 的维度 stride = 0（广播维）
inline DimArray broadcast_strides(std::span<const int64_t> shape,
                                  std::span<const int64_t> strides,
                                  size_t out_ndim) {
    DimArray eff;
    eff.resize(out_ndim);
    const size_t pad = out_ndim - shape.size();
    for (size_t i = 0; i < shape.size(); ++i)
        eff[pad + i] = (shape[i] == 1) ? 0 : strides[i];
    return eff;
}

// 广播驱动器：odometer 外层 + 尾部连续段内层（段内直接调线性 kernel）。
// 三个指针可以是不同类型（逻辑 op：两种输入 dtype + int8 输出）。
// 尾维任一侧为广播维（stride 0）时段长退化为 1 —— 正确但每元素一次 kernel
// 调用；矩阵+行向量这类常见形态走连续段快路径。
template <class AP, class BP, class OP, class Kernel>
void broadcast_drive(AP a, std::span<const int64_t> sa,
                     BP b, std::span<const int64_t> sb,
                     OP out, std::span<const int64_t> shape, Kernel&& k) {
    const size_t ndim = shape.size();
    size_t total = 1;
    for (int64_t d : shape) total *= size_t(d);

    // 尾部连续段：两侧有效 strides 与行主序一致的尾部维度
    size_t inner = 1;
    if (ndim > 0 && sa[ndim - 1] == 1 && sb[ndim - 1] == 1) {
        inner = size_t(shape[ndim - 1]);
        for (size_t d = ndim - 1; d-- > 0;) {
            if (sa[d] == int64_t(inner) && sb[d] == int64_t(inner) && shape[d] != 1)
                inner *= size_t(shape[d]);
            else if (sa[d] == int64_t(inner) && sb[d] == int64_t(inner))
                inner *= size_t(shape[d]);  // shape[d]==1 时并入无害
            else break;
        }
    }

    // 段外维度做 odometer，同时推进三个偏移
    size_t seg_dims = 0, acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        if (acc >= inner) break;  // inner 覆盖的尾部维度数
        acc *= size_t(shape[d]);
        ++seg_dims;
    }
    const size_t m = ndim - seg_dims;
    DimArray rad, ca, cb, co;
    rad.assign(shape.begin(), shape.begin() + m);
    ca.assign(sa.begin(), sa.begin() + m);
    cb.assign(sb.begin(), sb.begin() + m);
    co.resize(m);
    {
        int64_t o = int64_t(inner);
        for (size_t d = m; d-- > 0;) { co[d] = o; o *= shape[d]; }
    }
    DimArray coord;
    coord.resize(m);
    size_t off_a = 0, off_b = 0, off_o = 0;
    const size_t outer = total / inner;
    for (size_t it = 0; it < outer; ++it) {
        k(a + off_a, b + off_b, out + off_o, inner);
        for (size_t d = m; d-- > 0;) {
            ++coord[d];
            off_a += size_t(ca[d]); off_b += size_t(cb[d]); off_o += size_t(co[d]);
            if (coord[d] < rad[d]) break;
            coord[d] = 0;
            off_a -= size_t(ca[d]) * size_t(rad[d]);
            off_b -= size_t(cb[d]) * size_t(rad[d]);
            off_o -= size_t(co[d]) * size_t(rad[d]);
        }
    }
}

// ===== 二元 element-wise：T,T -> T（支持 NumPy 广播）=====
// Op 需提供: template <class T> static void kernel(const T* a, const T* b, T* out, size_t n);
template <class Op>
Tensor binary_op(const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor promoted_a, promoted_b, lin_a, lin_b;
    const Tensor* A = &a;
    const Tensor* B = &b;
    // 指针按需选择：三元表达式会因 lvalue/prvalue 混合隐式深拷贝已匹配的输入
    if (a.dtype() != res_type) { promoted_a = ops::Typecast(a, res_type); A = &promoted_a; }
    if (b.dtype() != res_type) { promoted_b = ops::Typecast(b, res_type); B = &promoted_b; }
    A = &linearize(*A, lin_a);
    B = &linearize(*B, lin_b);

    if (same_shape(A->shape(), B->shape())) {
        // 快路径：同形连续，单次线性 kernel
        Tensor result(A->shape(), res_type, Device::CPU);
        dispatch_dtype(res_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T>(
                static_cast<const T*>(A->data()),
                static_cast<const T*>(B->data()),
                static_cast<T*>(result.data()),
                A->numel());
        });
        return result;
    }

    DimArray out_shape;
    if (!broadcast_shapes(A->shape(), B->shape(), out_shape))
        throw std::runtime_error("binary op: shapes are not broadcastable");
    Tensor result(std::span<const int64_t>(out_shape), res_type, Device::CPU);
    const auto sa = broadcast_strides(A->shape(), A->strides(), out_shape.size());
    const auto sb = broadcast_strides(B->shape(), B->strides(), out_shape.size());
    dispatch_dtype(res_type, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        broadcast_drive(
            static_cast<const T*>(A->data()), std::span<const int64_t>(sa),
            static_cast<const T*>(B->data()), std::span<const int64_t>(sb),
            static_cast<T*>(result.data()), std::span<const int64_t>(out_shape),
            [](const T* x, const T* y, T* o, size_t n) {
                Op::template kernel<T>(x, y, o, n);
            });
    });
    return result;
}

// dst 输出版本（等价于 NumPy 的 out=，dst 形状须等于广播后的形状）
template <class Op>
void binary_op(const Tensor& a, const Tensor& b, Tensor& dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    if (dst.dtype() != res_type)
        throw std::runtime_error("dst dtype must equal the promoted result dtype");
    require_contiguous(dst, "dst");
    Tensor promoted_a, promoted_b, lin_a, lin_b;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { promoted_a = ops::Typecast(a, res_type); A = &promoted_a; }
    if (b.dtype() != res_type) { promoted_b = ops::Typecast(b, res_type); B = &promoted_b; }
    A = &linearize(*A, lin_a);
    B = &linearize(*B, lin_b);

    if (same_shape(A->shape(), B->shape())) {
        if (!same_shape(dst.shape(), A->shape()))
            throw std::runtime_error("dst shape must equal the broadcast result shape");
        dispatch_dtype(res_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T>(
                static_cast<const T*>(A->data()),
                static_cast<const T*>(B->data()),
                static_cast<T*>(dst.data()),
                A->numel());
        });
        return;
    }

    DimArray out_shape;
    if (!broadcast_shapes(A->shape(), B->shape(), out_shape))
        throw std::runtime_error("binary op: shapes are not broadcastable");
    if (!same_shape(dst.shape(), out_shape))
        throw std::runtime_error("dst shape must equal the broadcast result shape");
    const auto sa = broadcast_strides(A->shape(), A->strides(), out_shape.size());
    const auto sb = broadcast_strides(B->shape(), B->strides(), out_shape.size());
    dispatch_dtype(res_type, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        broadcast_drive(
            static_cast<const T*>(A->data()), std::span<const int64_t>(sa),
            static_cast<const T*>(B->data()), std::span<const int64_t>(sb),
            static_cast<T*>(dst.data()), std::span<const int64_t>(out_shape),
            [](const T* x, const T* y, T* o, size_t n) {
                Op::template kernel<T>(x, y, o, n);
            });
    });
}

// ===== 逻辑比较：T,R -> INT8（支持 NumPy 广播）=====
// 输入类型不统一提升（kernel 内 PromotedType 比较，保留混合 dtype 零拷贝），
// 结果固定 INT8 表示布尔。
// Op 需提供: template <class T, class R> static void kernel(int8_t* out, const T* a, const R* b, size_t n);
template <class Op>
Tensor logical_op(const Tensor& a, const Tensor& b) {
    Tensor lin_a, lin_b;
    const Tensor* A = &linearize(a, lin_a);
    const Tensor* B = &linearize(b, lin_b);

    DimArray out_shape;
    if (same_shape(A->shape(), B->shape())) {
        out_shape.assign(A->shape().begin(), A->shape().end());
    } else if (!broadcast_shapes(A->shape(), B->shape(), out_shape)) {
        throw std::runtime_error("logical op: shapes are not broadcastable");
    }
    Tensor result(std::span<const int64_t>(out_shape), DataType::INT8, Device::CPU);
    int8_t* out_ptr = static_cast<int8_t*>(result.data());

    auto va = data_as_const_variant(A->dtype(), A->data());
    auto vb = data_as_const_variant(B->dtype(), B->data());
    std::visit([&](auto pa, auto pb) {
        using TA = std::remove_cv_t<std::remove_pointer_t<decltype(pa)>>;
        using TB = std::remove_cv_t<std::remove_pointer_t<decltype(pb)>>;
        if (same_shape(A->shape(), B->shape())) {
            Op::template kernel<TA, TB>(out_ptr, pa, pb, A->numel());
        } else {
            const auto sa = broadcast_strides(A->shape(), A->strides(), out_shape.size());
            const auto sb = broadcast_strides(B->shape(), B->strides(), out_shape.size());
            broadcast_drive(pa, std::span<const int64_t>(sa), pb,
                            std::span<const int64_t>(sb), out_ptr,
                            std::span<const int64_t>(out_shape),
                [](const TA* x, const TB* y, int8_t* o, size_t n) {
                    Op::template kernel<TA, TB>(o, x, y, n);
                });
        }
    }, va, vb);
    return result;
}

// 标量比较：右侧是 FLOAT32 单元素（与原 Fill(FLOAT32)+比较 的语义一致），
// 省掉全形状临时张量的分配与填充；kernel 重复读同一地址（L1 命中）。
template <class Op>
Tensor logical_op_scalar(const Tensor& a, float val) {
    Tensor storage;
    const Tensor* A = &linearize(a, storage);
    Tensor result(A->shape(), DataType::INT8, Device::CPU);
    int8_t* out_ptr = static_cast<int8_t*>(result.data());
    auto va = data_as_const_variant(A->dtype(), A->data());
    std::visit([&](auto pa) {
        using TA = std::remove_cv_t<std::remove_pointer_t<decltype(pa)>>;
        Op::template kernel_scalar<TA>(out_ptr, pa, val, A->numel());
    }, va);
    return result;
}

// ===== 标量 in-place / out-of-place =====
// Op 需提供: template <class T> static void kernel(T* a, float b, size_t n);
template <class Op>
void scalar_inplace(Tensor& a, float b) {
    require_contiguous(a, "in-place element-wise op");
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        Op::template kernel<T>(static_cast<T*>(a.data()), b, a.numel());
    });
}

template <class Op>
Tensor scalar_out(const Tensor& a, float b) {
    Tensor t = a.clone();
    scalar_inplace<Op>(t, b);
    return t;
}

// ===== unary：T -> R =====
// Op 需提供:
//   template <class T, class R> static void kernel(const T* src, R* dst, size_t n, Args...);
//   static DataType result_type(DataType in);             // 恒等或整型 -> FLOAT32
//   static constexpr bool identity_result;                // true = result_type 恒等（无提升路径）
template <class Op, class... Args>
Tensor unary_op(const Tensor& a, Args... args) {
    Tensor storage;
    const Tensor* A = &linearize(a, storage);
    const DataType rt = Op::result_type(A->dtype());
    Tensor result(A->shape(), rt, Device::CPU);
    if (rt == A->dtype()) {
        dispatch_dtype(rt, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T, T>(
                static_cast<const T*>(A->data()),
                static_cast<T*>(result.data()),
                A->numel(), args...);
        });
    } else if constexpr (Op::identity_result) {
        // 恒等 Op 不存在类型提升路径，运行时不可达
        throw std::runtime_error("unary op: unreachable type promotion path");
    } else {
        // 唯一的提升路径：整型 -> FLOAT32（如 sin/int32）
        dispatch_dtype(A->dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T, float32>(
                static_cast<const T*>(A->data()),
                static_cast<float32*>(result.data()),
                A->numel(), args...);
        });
    }
    return result;
}

// Op 需提供原地 kernel: template <class T, class R> static void kernel(const T* src, R* dst, size_t n, Args...);
template <class Op, class... Args>
void unary_inplace(Tensor& a, Args... args) {
    require_contiguous(a, "in-place element-wise op");
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        Op::template kernel<T, T>(
            static_cast<T*>(a.data()),
            static_cast<T*>(a.data()),
            a.numel(), args...);
    });
}

}  // namespace ufunc

}  // namespace ops
