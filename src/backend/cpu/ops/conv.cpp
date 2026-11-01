// Device::CPU 的 2-D 卷积。
//
// 直接卷积：一个输出元素 = 一个 (c, i, j) 三元组的内积。im2col + GEMM 的吞吐
// 更高，代价是先把输入复制 K 倍（K = C·kh·kw）；推理侧的卷积核通常只有几十到
// 几百个 tap，直接卷积不用暂存、也不用在 GEMM 前把整块输入重排一遍。
//
// 累加用 compute_type_t<T>：半精度输入在半精度里累加会丢有效位，所以拓宽到
// float 再写回。
#include <cstddef>
#include <cstdint>
#include <variant>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"

namespace ops {

namespace {

template <typename T>
void conv2d_kernel(const T* in, const T* w, const T* bias, T* out, const Conv2dShape& s) {
    using Acc = compute_type_t<T>;

    const int64_t out_plane = s.out_h * s.out_w;
    const int64_t in_plane  = s.in_h * s.in_w;
    const int64_t taps      = s.in_ch * s.kh * s.kw;

    // 每个 (n, oc) 是一个互不相关的输出通道平面；内层按 (c, i, j) 顺序走权重，
    // 权重指针全程只向前。
    #pragma omp parallel for collapse(2) if (s.numel() > 4096)
    for (int64_t n = 0; n < s.batch; ++n) {
        for (int64_t oc = 0; oc < s.out_ch; ++oc) {
            const T* w_oc = w + oc * taps;
            const Acc b = bias != nullptr ? static_cast<Acc>(bias[oc]) : Acc(0);
            T* out_plane_ptr = out + (n * s.out_ch + oc) * out_plane;

            for (int64_t oy = 0; oy < s.out_h; ++oy) {
                for (int64_t ox = 0; ox < s.out_w; ++ox) {
                    Acc acc = b;
                    for (int64_t c = 0; c < s.in_ch; ++c) {
                        const T* in_c = in + (n * s.in_ch + c) * in_plane;
                        const T* w_c  = w_oc + c * s.kh * s.kw;
                        for (int64_t i = 0; i < s.kh; ++i) {
                            const int64_t y = oy * s.stride + i - s.padding;
                            if (y < 0 || y >= s.in_h) continue;
                            const T* in_row = in_c + y * s.in_w;
                            const T* w_row  = w_c + i * s.kw;
                            for (int64_t j = 0; j < s.kw; ++j) {
                                const int64_t x = ox * s.stride + j - s.padding;
                                if (x < 0 || x >= s.in_w) continue;
                                acc += static_cast<Acc>(in_row[x]) * static_cast<Acc>(w_row[j]);
                            }
                        }
                    }
                    out_plane_ptr[oy * s.out_w + ox] = static_cast<T>(acc);
                }
            }
        }
    }
}

}  // namespace

Tensor Conv2dImpl<Device::CPU>::execute(const Tensor& input, const Tensor& weight,
                                        const Tensor* bias, const Conv2dShape& shape) {
    Tensor result({shape.batch, shape.out_ch, shape.out_h, shape.out_w}, input.dtype(),
                  Device::CPU);
    const void* bias_ptr = bias != nullptr ? bias->data() : nullptr;

    dispatch_dtype(input.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        conv2d_kernel<T>(static_cast<const T*>(input.data()), static_cast<const T*>(weight.data()),
                         static_cast<const T*>(bias_ptr), static_cast<T*>(result.data()), shape);
    });
    return result;
}

}  // namespace ops
