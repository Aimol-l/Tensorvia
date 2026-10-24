#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"

namespace ops {

// index_select：沿 dim 用 1D idx 选取。输入已 contiguous，布局 [outer, axis_len, inner]。
// 每个 (outer, k) 沿 inner 做一段连续拷贝（embedding 场景 inner=特征维，天然友好）。
template <typename T, typename I>
void index_select_kernel(const T* in, const I* idx, T* out,
                         int64_t outer, int64_t axis_len, int64_t inner, int64_t idx_len) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t k = 0; k < idx_len; ++k) {
            const T* src = in + (o * axis_len + static_cast<int64_t>(idx[k])) * inner;
            T* dst = out + (o * idx_len + k) * inner;
            for (int64_t inn = 0; inn < inner; ++inn)
                dst[inn] = src[inn];
        }
}

// gather：out shape = index.shape（与 input 同秩），out[c] = input[c 但 dim 位置换成 index[c]]
// 非 dim 维的坐标随输出扁平索引按 odometer 递增，用 base 偏移累积其 in_stride 贡献，
// 逐元素只读 idx[i] 并乘 dim_stride —— 消除原逐元素 div/mod 解码。
template <typename T, typename I>
void gather_kernel(const T* in, const I* idx, T* out, size_t total,
                   const int64_t* out_shape, const int64_t* in_shape, int64_t dim, size_t ndim) {
    int64_t in_stride[kMaxTensorDims];
    {
        int64_t acc = 1;
        for (size_t d = ndim; d-- > 0;) { in_stride[d] = acc; acc *= in_shape[d]; }
    }
    const int64_t dim_stride = in_stride[dim];

    #pragma omp parallel
    {
        size_t begin = 0, end = total;
#ifdef _OPENMP
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t chunk = (total + size_t(nt) - 1) / size_t(nt);
        begin = size_t(tid) * chunk;
        end = begin + chunk < total ? begin + chunk : total;
#endif
        if (begin < end) {
            int64_t coord[kMaxTensorDims] = {};
            int64_t base = 0;
            // 一次性把线程起点解码成坐标 + base（每线程 ndim 次除，摊销后忽略）
            {
                size_t rem = begin;
                for (size_t d = ndim; d-- > 0;) {
                    coord[d] = static_cast<int64_t>(rem % size_t(out_shape[d]));
                    rem /= size_t(out_shape[d]);
                    if (d != size_t(dim)) base += coord[d] * in_stride[d];
                }
            }
            for (size_t i = begin; i < end; ++i) {
                out[i] = in[base + static_cast<int64_t>(idx[i]) * dim_stride];
                for (size_t d = ndim; d-- > 0;) {
                    if (d == size_t(dim)) {
                        if (++coord[d] < out_shape[d]) break;
                        coord[d] = 0;
                    } else {
                        ++coord[d];
                        base += in_stride[d];
                        if (coord[d] < out_shape[d]) break;
                        coord[d] = 0;
                        base -= in_stride[d] * out_shape[d];
                    }
                }
            }
        }
    }
}

// 索引越界校验（串行，进入并行 kernel 前完成，避免并行区抛异常）
template <typename I>
void check_index_range(const I* idx, int64_t len, int64_t axis_len, const char* op) {
    for (int64_t k = 0; k < len; ++k)
        // Compare as double: the software bfloat16/float16 types convert to
        // both float and double, so `idx[k] < 0` would be ambiguous.
        if (static_cast<double>(idx[k]) < 0.0 || static_cast<int64_t>(idx[k]) >= axis_len)
            throw std::runtime_error(std::string(op) + ": index out of range");
}

Tensor IndexSelectImpl<Device::CPU>::execute(const Tensor& input, int dim, const Tensor& index) {
    Tensor li, lx;
    const Tensor* in = &ufunc::linearize(input, li);
    const Tensor* idx = &ufunc::linearize(index, lx);

    const auto in_shape = in->shape();
    const size_t ndim = in_shape.size();
    const int64_t idx_len = static_cast<int64_t>(idx->numel());

    std::vector<int64_t> out_shape_v;
    out_shape_v.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i)
        out_shape_v.push_back(i == size_t(dim) ? idx_len : in_shape[i]);
    Tensor result(out_shape_v, input.dtype(), Device::CPU);

    int64_t outer = 1, inner = 1, axis_len = in_shape[dim];
    for (int64_t i = 0; i < dim; ++i) outer *= in_shape[i];
    for (size_t i = size_t(dim) + 1; i < ndim; ++i) inner *= in_shape[i];

    dispatch_dtype(index.dtype(), [&](auto iid) {
        using I = typename decltype(iid)::type;
        const I* idx_p = static_cast<const I*>(idx->data());
        check_index_range<I>(idx_p, idx_len, axis_len, "index_select");
        dispatch_dtype(input.dtype(), [&](auto tid) {
            using T = typename decltype(tid)::type;
            index_select_kernel<T, I>(static_cast<const T*>(in->data()), idx_p,
                                      static_cast<T*>(result.data()),
                                      outer, axis_len, inner, idx_len);
        });
    });
    return result;
}

Tensor GatherImpl<Device::CPU>::execute(const Tensor& input, int dim, const Tensor& index) {
    Tensor li, lx;
    const Tensor* in = &ufunc::linearize(input, li);
    const Tensor* idx = &ufunc::linearize(index, lx);

    const auto in_shape = in->shape();
    const auto idx_shape = idx->shape();
    const size_t ndim = in_shape.size();
    if (idx_shape.size() != ndim)
        throw std::runtime_error("gather: index must have the same rank as input");
    const int64_t axis_len = in_shape[dim];

    // 非 dim 维度 index 尺寸不得超出 input（否则坐标解码会越界读）
    for (size_t d = 0; d < ndim; ++d)
        if (d != size_t(dim) && idx_shape[d] > in_shape[d])
            throw std::runtime_error("gather: index size exceeds input along non-dim axes");

    std::vector<int64_t> out_shape_v(idx_shape.begin(), idx_shape.end());
    Tensor result(out_shape_v, input.dtype(), Device::CPU);

    dispatch_dtype(index.dtype(), [&](auto iid) {
        using I = typename decltype(iid)::type;
        const I* idx_p = static_cast<const I*>(idx->data());
        check_index_range<I>(idx_p, static_cast<int64_t>(idx->numel()), axis_len, "gather");
        dispatch_dtype(input.dtype(), [&](auto tid) {
            using T = typename decltype(tid)::type;
            gather_kernel<T, I>(static_cast<const T*>(in->data()), idx_p,
                                static_cast<T*>(result.data()), result.numel(),
                                out_shape_v.data(), in_shape.data(), dim, ndim);
        });
    });
    return result;
}

}  // namespace ops
