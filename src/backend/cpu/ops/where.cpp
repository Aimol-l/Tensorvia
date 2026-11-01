#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"
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

    auto vc = data_as_const_variant(in[0].dtype(), in[0].data());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* ap = static_cast<const T*>(in[1].data());
        const T* bp = static_cast<const T*>(in[2].data());
        T* op = static_cast<T*>(result.data());
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            if (iter::linearizable(lay)) {
                where_kernel<C, T>(pc, ap, bp, op, lay.numel);
                return;
            }
            iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
                where_kernel<C, T>(pc + off[0], ap + off[1], bp + off[2], op + oo, n);
            });
        }, vc);
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

    auto vc = data_as_const_variant(in[0].dtype(), in[0].data());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* ap = static_cast<const T*>(in[1].data());
        T* op = static_cast<T*>(result.data());
        const T bv = static_cast<T>(b);
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            if (iter::linearizable(lay)) {
                where_kernel_scalar_b<C, T>(pc, ap, bv, op, lay.numel);
                return;
            }
            iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
                where_kernel_scalar_b<C, T>(pc + off[0], ap + off[1], bv, op + oo, n);
            });
        }, vc);
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

    auto vc = data_as_const_variant(in[0].dtype(), in[0].data());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* bp = static_cast<const T*>(in[1].data());
        T* op = static_cast<T*>(result.data());
        const T av = static_cast<T>(a);
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            if (iter::linearizable(lay)) {
                where_kernel_scalar_a<C, T>(pc, av, bp, op, lay.numel);
                return;
            }
            iter::drive(lay, [&](std::span<const int64_t> off, int64_t oo, size_t n) {
                where_kernel_scalar_a<C, T>(pc + off[0], av, bp + off[1], op + oo, n);
            });
        }, vc);
    });
    return result;
}

}  // namespace ops
