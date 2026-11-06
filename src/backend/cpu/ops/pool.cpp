// Device::CPU 的 2-D 池化（Max / Avg）。
//
// 一个输出元素 = 输入窗口的一次扫描：Max 取真值最大、Avg 求和后除以窗口元素
// 数。padding 区域不是真值，不参与 Max；Avg 的 pad 计 0 且计入分母
// （count_include_pad，与 PyTorch 默认一致）。窗口整体越界（输出 0）时填 0。
//
// 累加用 compute_type_t<T>：半精度在半精度里累加会丢有效位，拓宽到 float/double
// 再写回。每个 (n, c) 输出平面互不相关，按它并行。
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"

namespace ops {

namespace {

template <typename T, bool kAvg>
void pool2d_kernel(const T* in, T* out, const Pool2dShape& s) {
    using Acc = compute_type_t<T>;

    const int64_t out_plane = s.out_h * s.out_w;
    const int64_t in_plane  = s.in_h * s.in_w;

    #pragma omp parallel for collapse(2) if (s.numel() > 4096)
    for (int64_t n = 0; n < s.batch; ++n) {
        for (int64_t c = 0; c < s.channels; ++c) {
            const T* in_plane_ptr = in + (n * s.channels + c) * in_plane;
            T* out_plane_ptr = out + (n * s.channels + c) * out_plane;

            for (int64_t oy = 0; oy < s.out_h; ++oy) {
                for (int64_t ox = 0; ox < s.out_w; ++ox) {
                    Acc acc   = kAvg ? Acc(0) : std::numeric_limits<Acc>::lowest();
                    int64_t count = 0;
                    for (int64_t i = 0; i < s.kh; ++i) {
                        const int64_t y = oy * s.stride_h + i - s.padding;
                        if (y < 0 || y >= s.in_h) continue;
                        const T* in_row = in_plane_ptr + y * s.in_w;
                        for (int64_t j = 0; j < s.kw; ++j) {
                            const int64_t x = ox * s.stride_w + j - s.padding;
                            if (x < 0 || x >= s.in_w) continue;
                            const Acc v = static_cast<Acc>(in_row[x]);
                            if constexpr (kAvg) {
                                acc += v;
                            } else {
                                acc = std::max(acc, v);
                            }
                            ++count;
                        }
                    }
                    // 窗口整体越界时没有真值：两种池化都填 0（Avg 也免了除零）。
                    out_plane_ptr[oy * s.out_w + ox] =
                        count == 0 ? T(0)
                                   : static_cast<T>(kAvg ? acc / static_cast<Acc>(s.kh * s.kw)
                                                         : acc);
                }
            }
        }
    }
}

Tensor pool2d_dispatch(const Tensor& input, const Pool2dShape& shape, bool average) {
    Tensor result({shape.batch, shape.channels, shape.out_h, shape.out_w}, input.dtype(),
                  Device::CPU);
    dispatch_dtype(input.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        if (average)
            pool2d_kernel<T, true>(static_cast<const T*>(input.data()),
                                   static_cast<T*>(result.data()), shape);
        else
            pool2d_kernel<T, false>(static_cast<const T*>(input.data()),
                                    static_cast<T*>(result.data()), shape);
    });
    return result;
}

}  // namespace

Tensor MaxPool2dImpl<Device::CPU>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_dispatch(input, shape, false);
}

Tensor AvgPool2dImpl<Device::CPU>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_dispatch(input, shape, true);
}

}  // namespace ops
