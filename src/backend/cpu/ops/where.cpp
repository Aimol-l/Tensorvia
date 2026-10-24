#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/ops.hpp"

namespace ops {

// cond 判真规则：非零为真（与 NumPy where 一致，兼容任意 dtype 的 cond）
template <typename C, typename T>
void where_kernel(const C* cond, const T* a, const T* b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = (cond[i] != C(0)) ? a[i] : b[i];
}

template <typename C, typename T>
void where_kernel_scalar_b(const C* cond, const T* a, T b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = (cond[i] != C(0)) ? a[i] : b;
}

template <typename C, typename T>
void where_kernel_scalar_a(const C* cond, T a, const T* b, T* out, size_t n) {
    #pragma omp parallel for if (n > 4096)
    for (size_t i = 0; i < n; ++i)
        out[i] = (cond[i] != C(0)) ? a : b[i];
}

// 3 路广播（cond/a/b → out）：flat 输出索引解码到坐标再累加偏移。
// 维数上限 kMaxTensorDims=5，每元素 O(ndim) 解码开销可忽略；广播是 where 的稀有路径。
template <typename C, typename T>
void where_broadcast(const C* c, std::span<const int64_t> sc,
                     const T* a, std::span<const int64_t> sa,
                     const T* b, std::span<const int64_t> sb,
                     T* o, std::span<const int64_t> shape) {
    const size_t ndim = shape.size();
    size_t total = 1;
    for (int64_t d : shape) total *= size_t(d);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        size_t rem = i, off_c = 0, off_a = 0, off_b = 0;
        for (size_t d = ndim; d-- > 0;) {
            const size_t idx = rem % size_t(shape[d]);
            rem /= size_t(shape[d]);
            off_c += idx * size_t(sc[d]);
            off_a += idx * size_t(sa[d]);
            off_b += idx * size_t(sb[d]);
        }
        o[i] = (c[off_c] != C(0)) ? a[off_a] : b[off_b];
    }
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor pa, pb, lc, la, lb;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }
    const Tensor* C = &ufunc::linearize(cond, lc);
    A = &ufunc::linearize(*A, la);
    B = &ufunc::linearize(*B, lb);

    auto vc = data_as_const_variant(C->dtype(), C->data());
    if (ufunc::same_shape(C->shape(), A->shape()) && ufunc::same_shape(A->shape(), B->shape())) {
        Tensor result(A->shape(), res_type, Device::CPU);
        dispatch_dtype(res_type, [&](auto tid) {
            using T = typename decltype(tid)::type;
            const T* ap = static_cast<const T*>(A->data());
            const T* bp = static_cast<const T*>(B->data());
            T* op = static_cast<T*>(result.data());
            std::visit([&](auto pc) {
                using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
                where_kernel<C, T>(pc, ap, bp, op, A->numel());
            }, vc);
        });
        return result;
    }

    DimArray out_shape, tmp;
    if (!ufunc::broadcast_shapes(C->shape(), A->shape(), tmp) ||
        !ufunc::broadcast_shapes(std::span<const int64_t>(tmp), B->shape(), out_shape))
        throw std::runtime_error("where: cond/a/b shapes are not broadcastable");
    Tensor result(std::span<const int64_t>(out_shape), res_type, Device::CPU);
    const auto sc = ufunc::broadcast_strides(C->shape(), C->strides(), out_shape.size());
    const auto sa = ufunc::broadcast_strides(A->shape(), A->strides(), out_shape.size());
    const auto sb = ufunc::broadcast_strides(B->shape(), B->strides(), out_shape.size());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* ap = static_cast<const T*>(A->data());
        const T* bp = static_cast<const T*>(B->data());
        T* op = static_cast<T*>(result.data());
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            where_broadcast<C, T>(pc, std::span<const int64_t>(sc),
                                  ap, std::span<const int64_t>(sa),
                                  bp, std::span<const int64_t>(sb),
                                  op, std::span<const int64_t>(out_shape));
        }, vc);
    });
    return result;
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, const Tensor& a, float b) {
    const DataType res_type = compute_type(a.dtype(), DataType::FLOAT32);
    Tensor pa, lc, la;
    const Tensor* A = &a;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    const Tensor* C = &ufunc::linearize(cond, lc);
    A = &ufunc::linearize(*A, la);

    auto vc = data_as_const_variant(C->dtype(), C->data());
    if (ufunc::same_shape(C->shape(), A->shape())) {
        Tensor result(A->shape(), res_type, Device::CPU);
        dispatch_dtype(res_type, [&](auto tid) {
            using T = typename decltype(tid)::type;
            const T* ap = static_cast<const T*>(A->data());
            T* op = static_cast<T*>(result.data());
            const T bv = static_cast<T>(b);
            std::visit([&](auto pc) {
                using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
                where_kernel_scalar_b<C, T>(pc, ap, bv, op, A->numel());
            }, vc);
        });
        return result;
    }

    DimArray out_shape;
    if (!ufunc::broadcast_shapes(C->shape(), A->shape(), out_shape))
        throw std::runtime_error("where: cond/a shapes are not broadcastable");
    Tensor result(std::span<const int64_t>(out_shape), res_type, Device::CPU);
    const auto sc = ufunc::broadcast_strides(C->shape(), C->strides(), out_shape.size());
    const auto sa = ufunc::broadcast_strides(A->shape(), A->strides(), out_shape.size());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* ap = static_cast<const T*>(A->data());
        T* op = static_cast<T*>(result.data());
        const T bv = static_cast<T>(b);
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            ufunc::broadcast_drive(pc, std::span<const int64_t>(sc),
                                   ap, std::span<const int64_t>(sa),
                                   op, std::span<const int64_t>(out_shape),
                [bv](const C* c, const T* x, T* o, size_t n) {
                    where_kernel_scalar_b<C, T>(c, x, bv, o, n);
                });
        }, vc);
    });
    return result;
}

Tensor WhereImpl<Device::CPU>::execute(const Tensor& cond, float a, const Tensor& b) {
    const DataType res_type = compute_type(DataType::FLOAT32, b.dtype());
    Tensor pb, lc, lb;
    const Tensor* B = &b;
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }
    const Tensor* C = &ufunc::linearize(cond, lc);
    B = &ufunc::linearize(*B, lb);

    auto vc = data_as_const_variant(C->dtype(), C->data());
    if (ufunc::same_shape(C->shape(), B->shape())) {
        Tensor result(B->shape(), res_type, Device::CPU);
        dispatch_dtype(res_type, [&](auto tid) {
            using T = typename decltype(tid)::type;
            const T* bp = static_cast<const T*>(B->data());
            T* op = static_cast<T*>(result.data());
            const T av = static_cast<T>(a);
            std::visit([&](auto pc) {
                using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
                where_kernel_scalar_a<C, T>(pc, av, bp, op, B->numel());
            }, vc);
        });
        return result;
    }

    DimArray out_shape;
    if (!ufunc::broadcast_shapes(C->shape(), B->shape(), out_shape))
        throw std::runtime_error("where: cond/b shapes are not broadcastable");
    Tensor result(std::span<const int64_t>(out_shape), res_type, Device::CPU);
    const auto sc = ufunc::broadcast_strides(C->shape(), C->strides(), out_shape.size());
    const auto sb = ufunc::broadcast_strides(B->shape(), B->strides(), out_shape.size());
    dispatch_dtype(res_type, [&](auto tid) {
        using T = typename decltype(tid)::type;
        const T* bp = static_cast<const T*>(B->data());
        T* op = static_cast<T*>(result.data());
        const T av = static_cast<T>(a);
        std::visit([&](auto pc) {
            using C = std::remove_cv_t<std::remove_pointer_t<decltype(pc)>>;
            ufunc::broadcast_drive(pc, std::span<const int64_t>(sc),
                                   bp, std::span<const int64_t>(sb),
                                   op, std::span<const int64_t>(out_shape),
                [av](const C* c, const T* x, T* o, size_t n) {
                    where_kernel_scalar_a<C, T>(c, av, x, o, n);
                });
        }, vc);
    });
    return result;
}

}  // namespace ops
