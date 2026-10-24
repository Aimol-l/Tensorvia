#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>
#include <utility>
#include <immintrin.h>
#include "_simd_math.h"

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"



namespace ops {

template <typename T>
void relu_kernel(const T* src, T* dst, size_t n) {
    #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] > T(0) ? src[i] : T(0);
    }
}

template <typename T, typename R>
void silu_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        const float* s = static_cast<const float*>(src);
        float* d = static_cast<float*>(dst);
        const size_t vec_end = n & ~size_t(7);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_end; i += 8) {
            _mm256_storeu_ps(d + i, silu_ps(_mm256_loadu_ps(s + i)));
        }
        for (size_t i = vec_end; i < n; ++i) {
            const float x = s[i];
            d[i] = x / (1.0f + std::exp(-x));
        }
    } else {
        #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<R>(PromotedType(src[i]) / (1 + std::exp(-PromotedType(src[i]))));
        }
    }
}
template <typename T, typename R = T>
void tanh_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        const float* s = static_cast<const float*>(src);
        float* d = static_cast<float*>(dst);
        const size_t vec_end = n & ~size_t(7);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_end; i += 8) {
            _mm256_storeu_ps(d + i, tanh_ps(_mm256_loadu_ps(s + i)));
        }
        for (size_t i = vec_end; i < n; ++i) {
            d[i] = std::tanh(s[i]);
        }
    } else {
        #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<R>(std::tanh(PromotedType(src[i])));
        }
    }
}

template <typename T, typename R>
void sigmoid_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        const float* s = static_cast<const float*>(src);
        float* d = static_cast<float*>(dst);
        const size_t vec_end = n & ~size_t(7);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_end; i += 8) {
            _mm256_storeu_ps(d + i, sigmoid_ps(_mm256_loadu_ps(s + i)));
        }
        for (size_t i = vec_end; i < n; ++i) {
            d[i] = 1.0f / (1.0f + std::exp(-s[i]));
        }
    } else {
        #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<R>(1 / (1 + std::exp(-PromotedType(src[i]))));
        }
    }
}
template <typename T, typename R>
void softmax_kernel(const T* src, R* res_ptr, size_t outer_size, size_t axis_size, size_t inner_size) {
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        if (inner_size == 1) {
            // f32 连续轴快路径：AVX2 向量化 max / exp-sum / normalize
            #pragma omp parallel for
            for (size_t o = 0; o < outer_size; ++o) {
                const float* x = static_cast<const float*>(src) + o * axis_size;
                float* y = static_cast<float*>(res_ptr) + o * axis_size;

                float max_val = -std::numeric_limits<float>::infinity();
                size_t i = 0;
                if (axis_size >= 8) {
                    __m256 vmax = _mm256_loadu_ps(x);
                    for (i = 8; i + 8 <= axis_size; i += 8)
                        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
                    max_val = hmax_ps(vmax);
                }
                for (; i < axis_size; ++i)
                    max_val = std::max(max_val, x[i]);

                const __m256 vmax = _mm256_set1_ps(max_val);
                __m256 sum = _mm256_setzero_ps();
                for (i = 0; i + 8 <= axis_size; i += 8) {
                    const __m256 e = exp_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmax));
                    sum = _mm256_add_ps(sum, e);
                    _mm256_storeu_ps(y + i, e);
                }
                float exp_sum = hsum_ps(sum);
                for (; i < axis_size; ++i) {
                    const float e = std::exp(x[i] - max_val);
                    exp_sum += e;
                    y[i] = e;
                }

                const float inv = 1.0f / exp_sum;
                const __m256 vinv = _mm256_set1_ps(inv);
                for (i = 0; i + 8 <= axis_size; i += 8)
                    _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), vinv));
                for (; i < axis_size; ++i)
                    y[i] *= inv;
            }
            return;
        }
    }
    #pragma omp parallel for
    for (size_t outer_index = 0; outer_index < outer_size; ++outer_index) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            for (size_t inner_index = 0; inner_index < inner_size; ++inner_index) {
                R max_val = -std::numeric_limits<R>::infinity();
                for (size_t k = 0; k < axis_size; ++k) {
                    R curr_val = src[outer_index * axis_size * inner_size + k * inner_size + inner_index];
                    max_val = max_val > curr_val ? max_val : curr_val;
                }
                float exp_sum = 0;
                for (size_t k = 0; k < axis_size; ++k) {
                    R curr_val = static_cast<R>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index]) - max_val;
                    exp_sum += std::exp(float(curr_val));
                }
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t pos = outer_index * axis_size * inner_size + k * inner_size + inner_index;
                    res_ptr[pos] = R(std::exp(static_cast<float>(T(src[pos]) - T(max_val))) / exp_sum);
                }
            }
        } else {
            for (size_t inner_index = 0; inner_index < inner_size; ++inner_index) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t k = 0; k < axis_size; ++k) {
                    max_val = std::max(max_val, static_cast<float>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index]));
                }
                float exp_sum = 0;
                for (size_t k = 0; k < axis_size; ++k) {
                    exp_sum += std::exp(static_cast<float>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index] - max_val));
                }
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t pos = outer_index * axis_size * inner_size + k * inner_size + inner_index;
                    res_ptr[pos] = std::exp(static_cast<float>(src[pos] - max_val)) / exp_sum;
                }
            }
        }
    }
}
// log_softmax：out = x - (max + log(Σ exp(x - max)))，数值稳定（先减 max 防上溢）。
// 结果 dtype 与 softmax 一致：整型 -> FLOAT32，否则保持输入 dtype。
template <typename T, typename R>
void log_softmax_kernel(const T* src, R* dst, size_t outer_size, size_t axis_size, size_t inner_size) {
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        if (inner_size == 1) {
            // f32 连续轴快路径：AVX2 向量化 max / exp-sum / x-lse（同 softmax）
            #pragma omp parallel for
            for (size_t o = 0; o < outer_size; ++o) {
                const float* x = static_cast<const float*>(src) + o * axis_size;
                float* y = static_cast<float*>(dst) + o * axis_size;

                float max_val = -std::numeric_limits<float>::infinity();
                size_t i = 0;
                if (axis_size >= 8) {
                    __m256 vmax = _mm256_loadu_ps(x);
                    for (i = 8; i + 8 <= axis_size; i += 8)
                        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
                    max_val = hmax_ps(vmax);
                }
                for (; i < axis_size; ++i)
                    max_val = std::max(max_val, x[i]);

                const __m256 vmax = _mm256_set1_ps(max_val);
                __m256 sum = _mm256_setzero_ps();
                for (i = 0; i + 8 <= axis_size; i += 8)
                    sum = _mm256_add_ps(sum, exp_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmax)));
                float exp_sum = hsum_ps(sum);
                for (; i < axis_size; ++i)
                    exp_sum += std::exp(x[i] - max_val);

                const float lse = max_val + std::log(exp_sum);
                const __m256 vlse = _mm256_set1_ps(lse);
                for (i = 0; i + 8 <= axis_size; i += 8)
                    _mm256_storeu_ps(y + i, _mm256_sub_ps(_mm256_loadu_ps(x + i), vlse));
                for (; i < axis_size; ++i)
                    y[i] = x[i] - lse;
            }
            return;
        }
    }
    #pragma omp parallel for
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            float m = -std::numeric_limits<float>::infinity();
            for (size_t k = 0; k < axis_size; ++k)
                m = std::max(m, static_cast<float>(src[o * axis_size * inner_size + k * inner_size + in]));
            float sumexp = 0.0f;
            for (size_t k = 0; k < axis_size; ++k)
                sumexp += std::exp(static_cast<float>(src[o * axis_size * inner_size + k * inner_size + in]) - m);
            const float lse = m + std::log(sumexp);
            for (size_t k = 0; k < axis_size; ++k)
                dst[o * axis_size * inner_size + k * inner_size + in] =
                    static_cast<R>(static_cast<float>(src[o * axis_size * inner_size + k * inner_size + in]) - lse);
        }
    }
}
// layer_norm：沿 axis..end 归一化（减均值除标准差），再融合 weight/bias 仿射。
// weight/bias 为归一化块形状（numel == 归一化长度），按行广播应用到每个块。
template <typename T, typename R>
void layer_norm_kernel(const T* src, const R* weight, const R* bias, R* dst,
                       size_t outer_size, size_t norm_size, float eps) {
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        // f32 快路径：AVX2 归约 mean/var + FMA 仿射
        const float* x = static_cast<const float*>(src);
        const float* w = static_cast<const float*>(weight);
        const float* b = static_cast<const float*>(bias);
        float* y = static_cast<float*>(dst);
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < outer_size; ++o) {
            const float* xr = x + o * norm_size;
            float* yr = y + o * norm_size;
            // mean
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= norm_size; i += 8)
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(xr + i));
            float sum = hsum_ps(acc);
            for (; i < norm_size; ++i) sum += xr[i];
            const float mean = sum / static_cast<float>(norm_size);
            // var = mean((x-mean)²)
            const __m256 vmean = _mm256_set1_ps(mean);
            __m256 vacc = _mm256_setzero_ps();
            for (i = 0; i + 8 <= norm_size; i += 8) {
                const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(xr + i), vmean);
                vacc = _mm256_add_ps(vacc, _mm256_mul_ps(d, d));
            }
            float var = hsum_ps(vacc);
            for (; i < norm_size; ++i) {
                const float d = xr[i] - mean;
                var += d * d;
            }
            var /= static_cast<float>(norm_size);
            // normalize + affine：y = (x-mean)/sqrt(var+eps)·w + b
            const float inv = 1.0f / std::sqrt(var + eps);
            const __m256 vinv = _mm256_set1_ps(inv);
            for (i = 0; i + 8 <= norm_size; i += 8) {
                const __m256 xv = _mm256_loadu_ps(xr + i);
                const __m256 xn = _mm256_mul_ps(_mm256_sub_ps(xv, vmean), vinv);
                _mm256_storeu_ps(yr + i,
                    _mm256_fmadd_ps(xn, _mm256_loadu_ps(w + i), _mm256_loadu_ps(b + i)));
            }
            for (; i < norm_size; ++i)
                yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (size_t o = 0; o < outer_size; ++o) {
        const T* x = src + o * norm_size;
        R* y = dst + o * norm_size;
        float mean = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) mean += static_cast<float>(x[i]);
        mean /= static_cast<float>(norm_size);
        float var = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) {
            const float d = static_cast<float>(x[i]) - mean;
            var += d * d;
        }
        var /= static_cast<float>(norm_size);
        const float inv = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < norm_size; ++i) {
            const float xn = (static_cast<float>(x[i]) - mean) * inv;
            y[i] = static_cast<R>(xn * static_cast<float>(weight[i]) + static_cast<float>(bias[i]));
        }
    }
}

// rms_norm：除以 sqrt(mean(x²)+eps)（无减均值），再融合 weight 缩放。
template <typename T, typename R>
void rms_norm_kernel(const T* src, const R* weight, R* dst,
                     size_t outer_size, size_t norm_size, float eps) {
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        // f32 快路径：AVX2 归约 mean(x²) + 乘 weight
        const float* x = static_cast<const float*>(src);
        const float* w = static_cast<const float*>(weight);
        float* y = static_cast<float*>(dst);
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < outer_size; ++o) {
            const float* xr = x + o * norm_size;
            float* yr = y + o * norm_size;
            // ms = mean(x²)
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= norm_size; i += 8) {
                const __m256 xv = _mm256_loadu_ps(xr + i);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, xv));
            }
            float ms = hsum_ps(acc);
            for (; i < norm_size; ++i) ms += xr[i] * xr[i];
            ms /= static_cast<float>(norm_size);
            const float inv = 1.0f / std::sqrt(ms + eps);
            const __m256 vinv = _mm256_set1_ps(inv);
            for (i = 0; i + 8 <= norm_size; i += 8) {
                const __m256 xv = _mm256_loadu_ps(xr + i);
                _mm256_storeu_ps(yr + i,
                    _mm256_mul_ps(_mm256_mul_ps(xv, vinv), _mm256_loadu_ps(w + i)));
            }
            for (; i < norm_size; ++i)
                yr[i] = xr[i] * inv * w[i];
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (size_t o = 0; o < outer_size; ++o) {
        const T* x = src + o * norm_size;
        R* y = dst + o * norm_size;
        float ms = 0.0f;
        for (size_t i = 0; i < norm_size; ++i) {
            const float v = static_cast<float>(x[i]);
            ms += v * v;
        }
        ms /= static_cast<float>(norm_size);
        const float inv = 1.0f / std::sqrt(ms + eps);
        for (size_t i = 0; i < norm_size; ++i)
            y[i] = static_cast<R>(static_cast<float>(x[i]) * inv * static_cast<float>(weight[i]));
    }
}

// ================================================================
// ufunc Op：只声明 kernel，调度/提升/物化由 ufunc 框架处理
struct ReluOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { relu_kernel<T>(src, dst, n); }
    static DataType result_type(DataType in) { return in; }
    static constexpr bool identity_result = true;
};
struct SiluOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { silu_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct TanhOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { tanh_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct SigmoidOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { sigmoid_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};

void ReluImpl<Device::CPU>::execute(Tensor& a)   { ufunc::unary_inplace<ReluOp>(a); }
void SiluImpl<Device::CPU>::execute(Tensor& a)   { ufunc::unary_inplace<SiluOp>(a); }
void TanhImpl<Device::CPU>::execute(Tensor& a)   { ufunc::unary_inplace<TanhOp>(a); }
void SigmoidImpl<Device::CPU>::execute(Tensor& a) { ufunc::unary_inplace<SigmoidOp>(a); }

Tensor ReluImpl<Device::CPU>::execute(const Tensor& a)   { return ufunc::unary_op<ReluOp>(a); }
Tensor SiluImpl<Device::CPU>::execute(const Tensor& a)   { return ufunc::unary_op<SiluOp>(a); }
Tensor TanhImpl<Device::CPU>::execute(const Tensor& a)   { return ufunc::unary_op<TanhOp>(a); }
Tensor SigmoidImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<SigmoidOp>(a); }
Tensor SoftmaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    int dim = static_cast<int>(a.shape().size());
    if (axis < 0)   axis += dim;
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= a.shape(i);
    size_t axis_size = a.shape(axis);
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i)
        inner_size *= a.shape(i);
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor res(a.shape(), res_type, a.device());
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        using R = std::conditional_t<std::is_integral_v<T>, float32, T>;
        const T* a_ptr = static_cast<const T*>(a.data());
        R* res_ptr = static_cast<R*>(res.data());
        softmax_kernel<T, R>(a_ptr, res_ptr, outer_size, axis_size, inner_size);
    });
    return res;
}
Tensor LogSoftmaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    int dim = static_cast<int>(a.shape().size());
    if (axis < 0)   axis += dim;
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= a.shape(i);
    size_t axis_size = a.shape(axis);
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i)
        inner_size *= a.shape(i);
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor res(a.shape(), res_type, a.device());
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        using R = std::conditional_t<std::is_integral_v<T>, float32, T>;
        const T* a_ptr = static_cast<const T*>(a.data());
        R* res_ptr = static_cast<R*>(res.data());
        log_softmax_kernel<T, R>(a_ptr, res_ptr, outer_size, axis_size, inner_size);
    });
    return res;
}
Tensor LayerNormImpl<Device::CPU>::execute(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps) {
    auto a_shape = a.shape();
    int dim = static_cast<int>(a_shape.size());
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];

    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64) res_type = DataType::FLOAT32;
    Tensor res(a.shape(), res_type, Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        using R = std::conditional_t<std::is_integral_v<T>, float32, T>;
        const T* a_ptr = static_cast<const T*>(a.data());
        const R* w_ptr = static_cast<const R*>(weight.data());
        const R* b_ptr = static_cast<const R*>(bias.data());
        R* res_ptr = static_cast<R*>(res.data());
        layer_norm_kernel<T, R>(a_ptr, w_ptr, b_ptr, res_ptr, outer_size, norm_size, eps);
    });
    return res;
}
Tensor RmsNormImpl<Device::CPU>::execute(const Tensor& a, const Tensor& weight, int axis, float eps) {
    auto a_shape = a.shape();
    int dim = static_cast<int>(a_shape.size());
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];

    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64) res_type = DataType::FLOAT32;
    Tensor res(a.shape(), res_type, Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        using R = std::conditional_t<std::is_integral_v<T>, float32, T>;
        const T* a_ptr = static_cast<const T*>(a.data());
        const R* w_ptr = static_cast<const R*>(weight.data());
        R* res_ptr = static_cast<R*>(res.data());
        rms_norm_kernel<T, R>(a_ptr, w_ptr, res_ptr, outer_size, norm_size, eps);
    });
    return res;
}
// ================================================================
}  // namespace ops
