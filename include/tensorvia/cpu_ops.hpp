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

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor_iterator.hpp"

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

template <Device D> struct SortImpl;
template <Device D> struct TopkImpl;
template <> struct SortImpl<Device::CPU> {
    static SortResult execute(const Tensor& input, int dim, bool descending);
};
template <> struct TopkImpl<Device::CPU> {
    static TopkResult execute(const Tensor& input, int64_t k, int dim, bool largest);
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

// ================================================================
// element-wise ufunc 框架：
// 类型提升 / 非连续视图物化 / 输出分配 / dtype 分发统一在这里处理，
// 各算子只需提供类型特化 kernel（Op 结构体），不再复制粘贴调度代码。
// 这是 NumPy ufunc 分层的 CPU 版：框架 = 迭代器 + 类型解析，Op = inner loop。
namespace ufunc {

// 始终物化（sort/gather 等索引空间算子：内核按连续 buffer 的下标语义读写，
// 逐元素搬运的收益不可能高于一次拷贝）。元素级与归约算子改走 ops::iter 的策略。
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

// 广播规则与 strided 布局的唯一实现在 ops::iter（tensor_iterator.hpp）；这里
// 保留别名，让 CPU 内核与各后端继续按 ufunc:: 的名字取用同一份。
using ops::iter::broadcast_shapes;
using ops::iter::broadcast_strides;
using ops::iter::same_shape;

// ===== 二元 element-wise：T,T -> T（支持 NumPy 广播）=====
// Op 需提供: template <class T> static void kernel(const T* a, const T* b, T* out, size_t n);
template <class Op>
Tensor binary_op(const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor promoted_a, promoted_b;
    const Tensor* A = &a;
    const Tensor* B = &b;
    // 指针按需选择：三元表达式会因 lvalue/prvalue 混合隐式深拷贝已匹配的输入
    if (a.dtype() != res_type) { promoted_a = ops::Typecast(a, res_type); A = &promoted_a; }
    if (b.dtype() != res_type) { promoted_b = ops::Typecast(b, res_type); B = &promoted_b; }

    // 形状、两侧 stride 与"要不要物化"都由统一迭代层给出：视图不再无条件拷贝。
    const iter::PreparedInputs in({A, B}, "binary op");
    const iter::Layout& lay = in.layout();
    const std::span<const int64_t> out_shape(lay.shape);
    Tensor result(out_shape, res_type, Device::CPU);

    if (iter::linearizable(lay)) {
        // 快路径：同形连续，单次线性 kernel
        dispatch_dtype(res_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T>(
                static_cast<const T*>(in[0].data()),
                static_cast<const T*>(in[1].data()),
                static_cast<T*>(result.data()),
                lay.numel);
        });
        return result;
    }

    dispatch_dtype(res_type, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* ap = static_cast<const T*>(in[0].data());
        const T* bp = static_cast<const T*>(in[1].data());
        T* op = static_cast<T*>(result.data());
        iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
            Op::template kernel<T>(ap + off[0], bp + off[1], op + oo, n);
        });
    });
    return result;
}

// dst 输出版本（等价于 NumPy 的 out=，dst 形状须等于广播后的形状）
template <class Op>
void binary_op(const Tensor& a, const Tensor& b, Tensor& dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor promoted_a, promoted_b;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { promoted_a = ops::Typecast(a, res_type); A = &promoted_a; }
    if (b.dtype() != res_type) { promoted_b = ops::Typecast(b, res_type); B = &promoted_b; }

    const iter::PreparedInputs in({A, B}, "binary op");
    const iter::Layout& lay = in.layout();
    const std::span<const int64_t> out_shape(lay.shape);
    iter::validate_dst(dst, out_shape, res_type);

    if (iter::linearizable(lay)) {
        dispatch_dtype(res_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            Op::template kernel<T>(
                static_cast<const T*>(in[0].data()),
                static_cast<const T*>(in[1].data()),
                static_cast<T*>(dst.data()),
                lay.numel);
        });
        return;
    }

    dispatch_dtype(res_type, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* ap = static_cast<const T*>(in[0].data());
        const T* bp = static_cast<const T*>(in[1].data());
        T* op = static_cast<T*>(dst.data());
        iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
            Op::template kernel<T>(ap + off[0], bp + off[1], op + oo, n);
        });
    });
}

// ===== 逻辑比较：T,R -> BOOL（支持 NumPy 广播）=====
// 输入类型不统一提升（kernel 内 PromotedType 比较，保留混合 dtype 零拷贝），
// 结果固定 BOOL 表示布尔（1-byte，规范化为 0/1）。
// Op 需提供: template <class T, class R> static void kernel(uint8_t* out, const T* a, const R* b, size_t n);
template <class Op>
Tensor logical_op(const Tensor& a, const Tensor& b) {
    // 混合 dtype 零拷贝：两侧各自保留原 dtype，比较在 kernel 内的公共类型里做。
    const iter::PreparedInputs in({&a, &b}, "logical op");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), DataType::BOOL, Device::CPU);
    uint8_t* out_ptr = static_cast<uint8_t*>(result.data());

    auto va = data_as_const_variant(in[0].dtype(), in[0].data());
    auto vb = data_as_const_variant(in[1].dtype(), in[1].data());
    std::visit([&](auto pa, auto pb) {
        using TA = std::remove_cv_t<std::remove_pointer_t<decltype(pa)>>;
        using TB = std::remove_cv_t<std::remove_pointer_t<decltype(pb)>>;
        if (iter::linearizable(lay)) {
            Op::template kernel<TA, TB>(out_ptr, pa, pb, lay.numel);
            return;
        }
        iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
            Op::template kernel<TA, TB>(out_ptr + oo, pa + off[0], pb + off[1], n);
        });
    }, va, vb);
    return result;
}

// 标量比较：右侧是 FLOAT32 单元素（与原 Fill(FLOAT32)+比较 的语义一致），
// 省掉全形状临时张量的分配与填充；kernel 重复读同一地址（L1 命中）。
template <class Op>
Tensor logical_op_scalar(const Tensor& a, float val) {
    const iter::PreparedInputs in({&a}, "logical op");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), DataType::BOOL, Device::CPU);
    uint8_t* out_ptr = static_cast<uint8_t*>(result.data());
    auto va = data_as_const_variant(in[0].dtype(), in[0].data());
    std::visit([&](auto pa) {
        using TA = std::remove_cv_t<std::remove_pointer_t<decltype(pa)>>;
        if (iter::linearizable(lay)) {
            Op::template kernel_scalar<TA>(out_ptr, pa, val, lay.numel);
            return;
        }
        iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
            Op::template kernel_scalar<TA>(out_ptr + oo, pa + off[0], val, n);
        });
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
    const iter::PreparedInputs in({&a}, "unary op");
    const iter::Layout& lay = in.layout();
    const DataType rt = Op::result_type(in[0].dtype());
    Tensor result(in[0].shape(), rt, Device::CPU);
    const bool linear = iter::linearizable(lay);

    if (rt == in[0].dtype()) {
        dispatch_dtype(rt, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            const T* src = static_cast<const T*>(in[0].data());
            T* dst = static_cast<T*>(result.data());
            if (linear) {
                Op::template kernel<T, T>(src, dst, lay.numel, args...);
                return;
            }
            iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
                Op::template kernel<T, T>(src + off[0], dst + oo, n, args...);
            });
        });
    } else if constexpr (Op::identity_result) {
        // 恒等 Op 不存在类型提升路径，运行时不可达
        throw std::runtime_error("unary op: unreachable type promotion path");
    } else {
        // 唯一的提升路径：整型 -> FLOAT32（如 sin/int32）
        dispatch_dtype(in[0].dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            const T* src = static_cast<const T*>(in[0].data());
            float32* dst = static_cast<float32*>(result.data());
            if (linear) {
                Op::template kernel<T, float32>(src, dst, lay.numel, args...);
                return;
            }
            iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
                Op::template kernel<T, float32>(src + off[0], dst + oo, n, args...);
            });
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
