#pragma once

// ================================================================
// FR-18 / FR-19：统一 strided 迭代层。
//
// 这一层只做后端无关的三件事：
//   1. 广播规则（broadcast_shapes / broadcast_strides）—— 全库唯一一份；
//   2. Layout：广播后每个操作数的元素级 stride、视图 offset 与总元素数。
//      CPU 用它驱动分段循环，设备后端把它翻译成内核参数结构体；
//   3. launch_binary<Backend>：二元逐元素算子的发射流程（形状推导 → dst 校验
//      → 输出分配 → impl/context 取用 → 后端 launch）。
//
// 内核仍然只认识 typed pointer + stride：这一层不碰内核，也不决定内核怎么循环。
// 三个设备后端各自只剩一个策略类型 + 一个把 Layout 填进参数结构体的函数。

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"

namespace ops::iter {

// 一次迭代能带的操作数上限。二元算子用 2，三元（where）用 3；再多就该另立接口。
inline constexpr size_t kMaxOperands = 4;

// NumPy 广播规则：shape 右对齐逐维比较，相等或一方为 1 可广播。
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

// 对齐到输出 ndim 后的有效 strides：缺失维度或大小为 1 的维度 stride = 0（广播维）。
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

// span 没有 operator==（C++20 设计如此），统一走这个助手。
inline bool same_shape(std::span<const int64_t> a, std::span<const int64_t> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// 一个操作数在广播后的迭代布局。
struct OperandLayout {
    DimArray strides;      // 对齐到 ndim；广播维为 0（元素为单位）
    int64_t  offset = 0;   // 视图 offset（元素）；内核用基址 + offset 寻址
};

// 一次广播迭代的完整布局。不含 dtype 与数据指针——那两样由后端自己填进参数。
struct Layout {
    int32_t  ndim  = 0;
    int64_t  numel = 0;
    DimArray shape;                                        // 广播后的形状
    std::array<OperandLayout, kMaxOperands> operands{};
    size_t   n_operands    = 0;
    bool     broadcastless = false;   // 所有输入形状都等于广播后的形状
};

// 折叠所有输入的形状得到广播结果，并为每个操作数对齐一份 stride 表。
// 不可广播时抛 "<op>: shapes are not broadcastable"。
inline Layout build(std::span<const Tensor* const> inputs, std::string_view op) {
    if (inputs.size() > kMaxOperands)
        throw std::runtime_error(std::format(
            "{}: more than {} operands in one iteration", op, kMaxOperands));

    Layout lay;
    lay.n_operands = inputs.size();

    DimArray acc;
    for (const Tensor* t : inputs) {
        DimArray next;
        if (!broadcast_shapes(std::span<const int64_t>(acc), t->shape(), next))
            throw std::runtime_error(std::format("{}: shapes are not broadcastable", op));
        acc = next;
    }
    lay.shape = acc;
    lay.ndim  = static_cast<int32_t>(acc.size());

    int64_t total = 1;
    for (int64_t d : lay.shape) total *= d;
    lay.numel = total;

    bool all_full = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& t = *inputs[i];
        lay.operands[i].strides = broadcast_strides(t.shape(), t.strides(), lay.shape.size());
        lay.operands[i].offset  = t.offset();
        all_full = all_full && same_shape(t.shape(), lay.shape);
    }
    lay.broadcastless = all_full;
    return lay;
}

inline Layout build_binary(const Tensor& a, const Tensor& b, std::string_view op) {
    const Tensor* in[2] = {&a, &b};
    return build(std::span<const Tensor* const>(in, 2), op);
}

// dst 输出（NumPy 的 out=）的契约：连续、dtype 已提升、形状等于广播结果。
// 写入型算子不接受非连续视图——写穿共享内存的语义不在本层定义。
inline void validate_dst(const Tensor& dst, std::span<const int64_t> out_shape,
                         DataType out_dtype) {
    if (!dst.is_contiguous())
        throw std::runtime_error(
            "dst: non-contiguous view is not supported here; call to_contiguous() first");
    if (dst.dtype() != out_dtype)
        throw std::runtime_error("dst dtype must equal the promoted result dtype");
    if (!same_shape(dst.shape(), out_shape))
        throw std::runtime_error("dst shape must equal the broadcast result shape");
}

// 无需转换时零拷贝，否则物化一个 Typecast 临时量；storage 保活它到调用返回。
inline const Tensor& promote(const Tensor& t, DataType res, Tensor& storage) {
    if (t.dtype() == res) return t;
    storage = ops::Typecast(t, res);
    return storage;
}

// ================================================================
// 物化策略：什么时候"拷贝一次"比"带着 stride 逐段跑"便宜。
//
// 判定只看布局，不看后端——后端决定怎么使用这个判定（CPU 物化后走线性内核，
// 设备后端本来就能直接吃 stride）。
//
// 尾部连续段（trailing_linear_extent）是代价的代理：段长就是内层内核一次能处理
// 的元素数，段长为 1 时退化成每元素一次内核调用。
// ================================================================

inline constexpr size_t  kMaterializeMaxInner  = 4;      // 尾段短于此值
inline constexpr int64_t kMaterializeMinElements = 1024;  // 且总量大于此值才值得拷

// 所有输入操作数都按 1 递增、可以一次线性跨越的尾部元素数。
inline size_t trailing_linear_extent(const Layout& lay) {
    const size_t n = lay.shape.size();
    if (lay.numel == 0) return 0;
    if (n == 0) return 1;
    for (size_t i = 0; i < lay.n_operands; ++i)
        if (lay.operands[i].strides[n - 1] != 1) return 1;
    size_t inner = size_t(lay.shape[n - 1]);
    for (size_t d = n - 1; d-- > 0;) {
        for (size_t i = 0; i < lay.n_operands; ++i)
            if (lay.operands[i].strides[d] != int64_t(inner)) return inner;
        inner *= size_t(lay.shape[d]);
    }
    return inner;
}

// 所有操作数都与输出同形、且都能按行主序线性遍历 —— 单次内核即可覆盖全部元素。
// 尺寸为 1 的维度不参与比较：它只有一个元素，stride 取什么都无所谓。
inline bool linearizable(const Layout& lay) {
    if (!lay.broadcastless) return false;
    const size_t n = lay.shape.size();
    for (size_t i = 0; i < lay.n_operands; ++i) {
        int64_t expect = 1;
        for (size_t d = n; d-- > 0;) {
            if (lay.shape[d] == 1) continue;
            if (lay.operands[i].strides[d] != expect) return false;
            expect *= lay.shape[d];
        }
    }
    return true;
}

// 逐段遍历的固定开销相对拷贝成本不值得时，返回 true。
inline bool should_materialize(const Layout& lay) {
    if (linearizable(lay)) return false;
    return trailing_linear_extent(lay) < kMaterializeMaxInner &&
           lay.numel >= kMaterializeMinElements;
}

// 输入准备：按策略决定是否物化跨步操作数，并给出最终使用的布局。
//
// 物化只针对"自身视图不连续"的操作数——广播维（stride 0）本来就是重复读取，
// 拷贝成完整形状只会更亏。
//
// 持有所有物化结果，因此不可拷贝/移动：就地使用。
class PreparedInputs {
public:
    PreparedInputs(std::initializer_list<const Tensor*> inputs, std::string_view op)
        : m_op(op) {
        m_count = inputs.size();
        std::copy(inputs.begin(), inputs.end(), m_ptrs);
        refresh();
        if (!should_materialize(m_layout)) return;
        bool changed = false;
        for (size_t i = 0; i < m_count; ++i) {
            if (m_ptrs[i]->is_contiguous()) continue;
            m_storage[i] = m_ptrs[i]->contiguous();
            m_ptrs[i]     = &m_storage[i];
            changed       = true;
        }
        if (changed) refresh();
    }

    PreparedInputs(const PreparedInputs&)            = delete;
    PreparedInputs& operator=(const PreparedInputs&) = delete;

    const Layout& layout() const { return m_layout; }
    const Tensor& operator[](size_t i) const { return *m_ptrs[i]; }
    size_t size() const { return m_count; }

private:
    void refresh() {
        m_layout = build(std::span<const Tensor* const>(m_ptrs, m_count), m_op);
    }

    std::string_view m_op;
    Tensor       m_storage[kMaxOperands]{};
    const Tensor* m_ptrs[kMaxOperands]{};
    size_t       m_count = 0;
    Layout       m_layout;
};

// ================================================================
// 分段遍历：把 Layout 展开成"一段段连续内存"，每段交给线性内核处理。
//
// 回调签名：fn(输入偏移, 输出偏移, 段长)，偏移以元素为单位、相对各张量的
// Tensor::data()（已含视图 offset）。输出总是按行主序连续。
// ================================================================
template <class Fn>
void drive(const Layout& lay, Fn&& fn) {
    const size_t ndim = lay.shape.size();
    const size_t inner = trailing_linear_extent(lay);
    if (inner == 0) return;

    // inner 覆盖了尾部几维
    size_t seg_dims = 0, acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        if (acc >= inner) break;
        acc *= size_t(lay.shape[d]);
        ++seg_dims;
    }
    const size_t m = ndim - seg_dims;

    // 段外维：输入 stride 来自各操作数，输出 stride 是行主序。
    std::array<int64_t, kMaxTensorDims> o_stride{};
    {
        int64_t o = int64_t(inner);
        for (size_t d = m; d-- > 0;) {
            o_stride[d] = o;
            o *= lay.shape[d];
        }
    }

    std::array<int64_t, kMaxTensorDims> coord{};
    std::array<int64_t, kMaxOperands>   in_off{};
    int64_t out_off = 0;
    const size_t outer = size_t(lay.numel) / inner;

    for (size_t it = 0; it < outer; ++it) {
        fn(std::span<const int64_t>(in_off.data(), lay.n_operands), out_off, inner);
        for (size_t d = m; d-- > 0;) {
            ++coord[d];
            for (size_t i = 0; i < lay.n_operands; ++i) in_off[i] += lay.operands[i].strides[d];
            out_off += o_stride[d];
            if (coord[d] < lay.shape[d]) break;
            coord[d] = 0;
            for (size_t i = 0; i < lay.n_operands; ++i)
                in_off[i] -= lay.operands[i].strides[d] * lay.shape[d];
            out_off -= o_stride[d] * lay.shape[d];
        }
    }
}

// 二元逐元素算子的发射流程。Backend 需提供：
//   using Impl = ...;  using Ctx = ...;      // 张量 impl 与 context 类型
//   static constexpr Device device;          // 输出张量落在哪个设备
//   static constexpr std::string_view name;  // 错误信息里的后端前缀，如 "cuda"
//   static void activate(Ctx&);              // 让 ctx 成为调用线程的当前设备
//   static void launch(Ctx&, const Tensor& A, const Tensor& B, const Tensor& out,
//                      const std::shared_ptr<Impl>&, const std::shared_ptr<Impl>&,
//                      const std::shared_ptr<Impl>&, DataType out_dtype, Op,
//                      const Layout&);
template <class Backend, class Op>
Tensor launch_binary(Op op, const Tensor& a, const Tensor& b, DataType out_dtype,
                     Tensor* dst) {
    const Layout lay = build_binary(a, b, "binary op");
    const std::span<const int64_t> want(lay.shape);

    Tensor res;
    if (dst != nullptr) {
        validate_dst(*dst, want, out_dtype);
    } else {
        res = Tensor(want, out_dtype, Backend::device);
    }
    Tensor& out = dst != nullptr ? *dst : res;

    const auto ai = std::dynamic_pointer_cast<typename Backend::Impl>(a.get_impl());
    const auto bi = std::dynamic_pointer_cast<typename Backend::Impl>(b.get_impl());
    const auto oi = std::dynamic_pointer_cast<typename Backend::Impl>(out.get_impl());
    if (!ai || !bi || !oi)
        throw std::runtime_error(std::format("{} binary op: operands must live on the {} device",
                                             Backend::name, device_to_string(Backend::device)));
    auto ctx = std::dynamic_pointer_cast<typename Backend::Ctx>(ai->context());
    if (!ctx)
        throw std::runtime_error(std::format("{} binary op: missing {}Context",
                                             Backend::name, device_to_string(Backend::device)));

    Backend::activate(*ctx);
    Backend::launch(*ctx, a, b, out, ai, bi, oi, out_dtype, op, lay);
    return res;
}

}  // namespace ops::iter
