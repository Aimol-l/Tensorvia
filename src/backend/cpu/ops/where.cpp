#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/tensor_iterator.hpp"
#include "tensorvia/ops.hpp"

namespace ops {

// BOOL 输出会规范化：源 BOOL 字节（比如 2）选中真，写回为 1，绝不原样写字节。
template <typename T>
inline T normalized_bool(T v) {
    if constexpr (std::is_same_v<T, uint8_t>) return v != 0 ? uint8_t{1} : uint8_t{0};
    else return v;
}

// cond 判真规则：非零为真（与 NumPy where 一致，兼容任意 dtype 的 cond）
template <typename C, typename T>
void where_kernel(const C* cond, const T* a, const T* b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = normalized_bool((cond[i] != C(0)) ? a[i] : b[i]);
}

template <typename C, typename T>
void where_kernel_scalar_b(const C* cond, const T* a, T b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = normalized_bool((cond[i] != C(0)) ? a[i] : b);
}

template <typename C, typename T>
void where_kernel_scalar_a(const C* cond, T a, const T* b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = normalized_bool((cond[i] != C(0)) ? a : b[i]);
}

// 三路广播（cond/a/b → out）：布局与"是否物化"由统一迭代层给出，这里只按段
// 调内核。段内指针偏移，段外由 iter::drive 的 odometer 推进。
//
// 下面这些是函数模板，不是 execute 里的 lambda；cond 的 dtype 分支也走
// dispatch_dtype 的 switch，不用 data_as_const_variant + std::visit。两个改动
// 是同一件事：MSVC 在"模板 lambda 里再 std::visit"这个组合上直接 ICE
// （fatal error C1001，msc1.cpp:1589，崩在 _Variant_all_visit_results_same →
// std::invoke 的 noexcept 实参 → 实例化 visit lambda 的 operator()），而 where
// 是唯一把 lambda 叠到三层（dispatch_dtype λ → visit λ → drive λ）的地方。
// 摊平之后任何函数里最多只剩一层 lambda。
template <typename C, typename T>
void where_run(const iter::Layout& lay, const C* pc, const T* ap, const T* bp, T* op) {
    if (iter::linearizable(lay)) {
        where_kernel<C, T>(pc, ap, bp, op, lay.numel);
        return;
    }
    iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
        where_kernel<C, T>(pc + off[0], ap + off[1], bp + off[2], op + oo, n);
    });
}

template <typename C, typename T>
void where_run_b(const iter::Layout& lay, const C* pc, const T* ap, T b, T* op) {
    if (iter::linearizable(lay)) {
        where_kernel_scalar_b<C, T>(pc, ap, b, op, lay.numel);
        return;
    }
    iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
        where_kernel_scalar_b<C, T>(pc + off[0], ap + off[1], b, op + oo, n);
    });
}

template <typename C, typename T>
void where_run_a(const iter::Layout& lay, const C* pc, T a, const T* bp, T* op) {
    if (iter::linearizable(lay)) {
        where_kernel_scalar_a<C, T>(pc, a, bp, op, lay.numel);
        return;
    }
    iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
        where_kernel_scalar_a<C, T>(pc + off[0], a, bp + off[1], op + oo, n);
    });
}

// cond 的 dtype 分支。cond 只用来判真，但得按它自己的类型读字节，所以还是要分 9 支。
template <typename T>
void where_dispatch(DataType cond_type, const void* cond_data, const iter::Layout& lay,
                    const T* ap, const T* bp, T* op) {
    dispatch_dtype(cond_type, [&](auto cid) {
        using C = typename decltype(cid)::type;
        where_run(lay, static_cast<const C*>(cond_data), ap, bp, op);
    });
}

template <typename T>
void where_dispatch_b(DataType cond_type, const void* cond_data, const iter::Layout& lay,
                      const T* ap, T b, T* op) {
    dispatch_dtype(cond_type, [&](auto cid) {
        using C = typename decltype(cid)::type;
        where_run_b(lay, static_cast<const C*>(cond_data), ap, b, op);
    });
}

template <typename T>
void where_dispatch_a(DataType cond_type, const void* cond_data, const iter::Layout& lay,
                      T a, const T* bp, T* op) {
    dispatch_dtype(cond_type, [&](auto cid) {
        using C = typename decltype(cid)::type;
        where_run_a(lay, static_cast<const C*>(cond_data), a, bp, op);
    });
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor pa, pb;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }

    const iter::PreparedInputs in({&cond, A, B}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::CPU);

    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        where_dispatch(in[0].dtype(), in[0].data(), lay,
                       static_cast<const T*>(in[1].data()),
                       static_cast<const T*>(in[2].data()),
                       static_cast<T*>(result.data()));
    });
    return result;
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, const Tensor& a, float b) {
    const DataType res_type = scalar_promote_type(a.dtype());
    Tensor pa;
    const Tensor* A = &a;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }

    const iter::PreparedInputs in({&cond, A}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::CPU);

    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        where_dispatch_b(in[0].dtype(), in[0].data(), lay,
                         static_cast<const T*>(in[1].data()),
                         static_cast<T>(b),
                         static_cast<T*>(result.data()));
    });
    return result;
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, float a, const Tensor& b) {
    const DataType res_type = scalar_promote_type(b.dtype());
    Tensor pb;
    const Tensor* B = &b;
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }

    const iter::PreparedInputs in({&cond, B}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::CPU);

    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        where_dispatch_a(in[0].dtype(), in[0].data(), lay,
                         static_cast<T>(a),
                         static_cast<const T*>(in[1].data()),
                         static_cast<T*>(result.data()));
    });
    return result;
}

}  // namespace ops
