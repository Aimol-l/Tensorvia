#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cstdint>
#include <format>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"

namespace ops {

// sort/topk 都归结为「沿 dim 排一次下标」：把输入看成 [outer, n, inner]，
// 每个 (outer, inner) 列独立排序。下标用 std::stable_sort 的稳定次序保持相等
// 元素的原始相对位置（与 torch.sort(stable=True) 一致），内存访问沿 inner 步进。
template <typename T>
void sort_indices_kernel(const T* in, int64_t* out_idx, T* out_vals,
                         int64_t outer, int64_t n, int64_t inner,
                         bool descending, int64_t keep) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t col = 0; col < inner; ++col) {
            const T* base = in + (o * n) * inner + col;
            std::vector<int64_t> idx(static_cast<size_t>(n));
            std::iota(idx.begin(), idx.end(), int64_t(0));
            std::stable_sort(idx.begin(), idx.end(), [&](int64_t x, int64_t y) {
                const T vx = base[x * inner];
                const T vy = base[y * inner];
                return descending ? vy < vx : vx < vy;
            });
            T* dst = out_vals + (o * keep) * inner + col;
            int64_t* dst_idx = out_idx + (o * keep) * inner + col;
            for (int64_t i = 0; i < keep; ++i) {
                const int64_t src = idx[static_cast<size_t>(i)] * inner;
                dst[i * inner] = base[src];
                dst_idx[i * inner] = idx[static_cast<size_t>(i)];
            }
        }
    }
}

namespace {

// [outer, n, inner] 分解，以及 sort/topk 共用的输出形状。
void axis_layout(const Tensor& in, int dim, int64_t& outer, int64_t& n, int64_t& inner) {
    const auto shape = in.shape();
    outer = 1;
    n = shape[static_cast<size_t>(dim)];
    inner = 1;
    for (int i = 0; i < dim; ++i) outer *= shape[static_cast<size_t>(i)];
    for (size_t i = static_cast<size_t>(dim) + 1; i < shape.size(); ++i) inner *= shape[i];
}

std::vector<int64_t> axis_shape(const Tensor& in, int dim, int64_t n) {
    std::vector<int64_t> shape(in.shape().begin(), in.shape().end());
    shape[static_cast<size_t>(dim)] = n;
    return shape;
}

}  // namespace

SortResult SortImpl<Device::CPU>::execute(const Tensor& input, int dim, bool descending) {
    Tensor lin;
    const Tensor* in = &ufunc::linearize(input, lin);
    int64_t outer = 1, n = 1, inner = 1;
    axis_layout(*in, dim, outer, n, inner);

    const std::vector<int64_t> shape = axis_shape(*in, dim, n);
    Tensor values(shape, input.dtype(), Device::CPU);
    Tensor indices(shape, DataType::INT64, Device::CPU);

    dispatch_dtype(input.dtype(), [&](auto tid) {
        using T = typename decltype(tid)::type;
        sort_indices_kernel<T>(static_cast<const T*>(in->data()),
                               static_cast<int64_t*>(indices.data()),
                               static_cast<T*>(values.data()),
                               outer, n, inner, descending, n);
    });
    return SortResult{std::move(values), std::move(indices)};
}

TopkResult TopkImpl<Device::CPU>::execute(const Tensor& input, int64_t k, int dim, bool largest) {
    Tensor lin;
    const Tensor* in = &ufunc::linearize(input, lin);
    int64_t outer = 1, n = 1, inner = 1;
    axis_layout(*in, dim, outer, n, inner);
    if (k < 0 || k > n)
        throw std::runtime_error(std::format(
            "topk: k ({}) must be between 0 and the axis length ({})", k, n));

    const std::vector<int64_t> shape = axis_shape(*in, dim, k);
    Tensor values(shape, input.dtype(), Device::CPU);
    Tensor indices(shape, DataType::INT64, Device::CPU);

    dispatch_dtype(input.dtype(), [&](auto tid) {
        using T = typename decltype(tid)::type;
        sort_indices_kernel<T>(static_cast<const T*>(in->data()),
                               static_cast<int64_t*>(indices.data()),
                               static_cast<T*>(values.data()),
                               outer, n, inner, largest, k);
    });
    return TopkResult{std::move(values), std::move(indices)};
}

}  // namespace ops
