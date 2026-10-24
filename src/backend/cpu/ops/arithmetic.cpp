#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "_platform.h"
#include "_simd_math.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/ops.hpp"




namespace ops {

template <typename T>
void add_kernel(const T* RESTRICT a, const T* RESTRICT b, T* RESTRICT out, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

template <typename T> // T = int8_t ~ int64_t ,float16 ~ float64,bfloat16
void add_kernel(T* a, float b, size_t size) {
    const T b_cast = static_cast<T>(b); // 注意类型转换
    // 让编译器去优化速度
    for (size_t i = 0; i < size; ++i) {
        a[i] += b_cast;
    }
}
template <typename T>
void sub_kernel(const T* RESTRICT a, const T* RESTRICT b, T* RESTRICT out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<T>(a[i] - b[i]);  // 注意类型转换
    }
}

template <typename T>
void sub_kernel(T* a, float b, size_t size) {
    const T b_cast = static_cast<T>(b); // 注意类型转换
    for (size_t i = 0; i < size; ++i) {
        a[i] -= b_cast;  // 注意类型转换
    }
}
template <typename T>
void mul_kernel(const T* RESTRICT a, const T* RESTRICT b, T* RESTRICT out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<T>(a[i] * b[i]);  // 注意类型转换
    }
}

template <typename T>
void mul_kernel(T* a, float b, size_t size) {
    const T b_cast = static_cast<T>(b); // 注意类型转换
    for (size_t i = 0; i < size; ++i) {
        a[i] *= b_cast;  // 注意类型转换
    }
}
template <typename T>
void div_kernel(const T* RESTRICT a, const T* RESTRICT b, T* RESTRICT out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (b[i] != T(0))
            out[i] = static_cast<T>(a[i] / b[i]);  // 注意类型转换
        else
            out[i] = std::numeric_limits<T>::quiet_NaN();
    }
}
template <typename T>
void div_kernel(T* a, float b, size_t size) {
    const T b_cast = T(b); // 注意类型转换
    for (size_t i = 0; i < size; ++i) {
        a[i] /=b_cast;  // 注意类型转换
    }
}

template <typename T>
void abs_kernel(T* src_ptr, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        src_ptr[i] = abs(src_ptr[i]);
    }
}
template <typename T>
void clamp_kernel(const T* src_ptr, T* dst_ptr, size_t n, float min, float max) {
    T min_val = static_cast<T>(min);
    T max_val = static_cast<T>(max);
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        // dst_ptr[i] = src_ptr[i] > max_val ? max_val : (src_ptr[i] < min_val ? min_val : src_ptr[i]); // 使用的是左右严格比较，不采用左闭右开的区间格式，同步sycl的代码
        dst_ptr[i] = std::min(max_val, std::max(min_val, src_ptr[i]));
    }
}
template <typename T, typename R = T>
void sin_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float val = std::sinf(float(src_ptr[i]));
            dst_ptr[i] = static_cast<R>(val);
        } else {
            dst_ptr[i] = static_cast<R>(std::sin(src_ptr[i]));
        }
    }
}
template <typename T, typename R = T>
void cos_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float val = std::cosf(float(src_ptr[i]));
            dst_ptr[i] = static_cast<R>(val);
        } else {
            dst_ptr[i] = static_cast<R>(std::cos(src_ptr[i]));
        }
    }
}

template <typename T, typename R = T>
void tan_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float val = std::tanf(float(src_ptr[i]));
            dst_ptr[i] = static_cast<R>(val);
        } else {
            dst_ptr[i] = static_cast<R>(std::tan(src_ptr[i]));
        }
    }
}
template <typename T, typename R = T>
void exp_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
    if constexpr (std::is_same_v<T, float32> && std::is_same_v<R, float32>) {
        // f32 快路径：AVX2 多项式逼近，8 元素 SIMD + 标量尾
        const float* src = static_cast<const float*>(src_ptr);
        float* dst = static_cast<float*>(dst_ptr);
        const size_t vec_end = n & ~size_t(7);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < vec_end; i += 8) {
            _mm256_storeu_ps(dst + i, exp_ps(_mm256_loadu_ps(src + i)));
        }
        for (size_t i = vec_end; i < n; ++i) {
            dst[i] = std::exp(src[i]);
        }
    } else {
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
                float val = std::expf(float(src_ptr[i]));
                dst_ptr[i] = static_cast<R>(val);
            } else {
                dst_ptr[i] = static_cast<R>(std::exp(src_ptr[i]));
            }
        }
    }
}
template <typename T, typename R = T>
void erf_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float val = std::erff(float(src_ptr[i]));
            dst_ptr[i] = static_cast<R>(val);
        } else {
            dst_ptr[i] = static_cast<R>(std::erf(src_ptr[i]));
        }
    }
}
template <typename T, typename R = T>
void sqrt_kernel(const T* src_ptr, R* dst_ptr, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float val = std::sqrtf(float(src_ptr[i]));
            dst_ptr[i] = static_cast<R>(val);
        } else {
            dst_ptr[i] = static_cast<R>(std::sqrt(src_ptr[i]));
        }
    }
}
template <typename T, typename R = T>
void pow_kernel(const T* src_ptr, R* dst_ptr, size_t n, float val) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float r = std::powf(float(src_ptr[i]), val);
            dst_ptr[i] = static_cast<R>(r);
        } else {
            dst_ptr[i] = static_cast<R>(std::pow(src_ptr[i], T(val)));
        }
    }
}
template <typename T, typename R = T>
void log_kernel(const T* src_ptr, R* dst_ptr, size_t n, float val) {
    if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
        const float log_base = std::logf(val);  // 底数的对数是循环不变量
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = static_cast<R>(std::logf(float(src_ptr[i])) / log_base);
        }
    } else {
        const T log_base = std::log(T(val));
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            dst_ptr[i] = static_cast<R>(std::log(src_ptr[i]) / log_base);
        }
    }
}


// ================================================================
// ufunc Op：每个算子只声明 kernel（类型特化的 inner loop），
// 类型提升 / 视图物化 / 输出分配 / dtype 分发统一由 ufunc:: 框架处理。
// ================================================================

struct AddOp {
    template <class T>
    static void kernel(const T* a, const T* b, T* out, size_t n) { add_kernel(a, b, out, n); }
    template <class T>
    static void kernel(T* a, float b, size_t n) { add_kernel(a, b, n); }
};
struct SubOp {
    template <class T>
    static void kernel(const T* a, const T* b, T* out, size_t n) { sub_kernel(a, b, out, n); }
    template <class T>
    static void kernel(T* a, float b, size_t n) { sub_kernel(a, b, n); }
};
struct MulOp {
    template <class T>
    static void kernel(const T* a, const T* b, T* out, size_t n) { mul_kernel(a, b, out, n); }
    template <class T>
    static void kernel(T* a, float b, size_t n) { mul_kernel(a, b, n); }
};
struct DivOp {
    template <class T>
    static void kernel(const T* a, const T* b, T* out, size_t n) { div_kernel(a, b, out, n); }
    template <class T>
    static void kernel(T* a, float b, size_t n) { div_kernel(a, b, n); }
};
struct AbsOp {
    // 条件式取代原来的非限定 abs() 调用（后者对 float 走 <cstdlib> 的 int 版本）
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) {
        for (size_t i = 0; i < n; ++i)
            dst[i] = static_cast<R>(src[i] < T(0) ? -src[i] : src[i]);
    }
    static DataType result_type(DataType in) { return in; }
    static constexpr bool identity_result = true;
};
struct SinOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { sin_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct CosOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { cos_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct TanOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { tan_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct ExpOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { exp_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct SqrtOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { sqrt_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct ErfOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n) { erf_kernel<T, R>(src, dst, n); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct PowOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n, float val) { pow_kernel<T, R>(src, dst, n, val); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct LogOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n, float val) { log_kernel<T, R>(src, dst, n, val); }
    static DataType result_type(DataType in) {
        return in <= DataType::INT64 ? DataType::FLOAT32 : in;
    }
    static constexpr bool identity_result = false;
};
struct ClampOp {
    template <class T, class R = T>
    static void kernel(const T* src, R* dst, size_t n, float min, float max) { clamp_kernel<T>(src, dst, n, min, max); }
    static DataType result_type(DataType in) { return in; }
    static constexpr bool identity_result = true;
};

// ================================================================
// Impl::execute —— 全部一行转发进框架
// ================================================================
void AddImpl<Device::CPU>::execute(Tensor& a, float b)  { ufunc::scalar_inplace<AddOp>(a, b); }
void SubImpl<Device::CPU>::execute(Tensor& a, float b)  { ufunc::scalar_inplace<SubOp>(a, b); }
void MulImpl<Device::CPU>::execute(Tensor& a, float b)  { ufunc::scalar_inplace<MulOp>(a, b); }
void DivImpl<Device::CPU>::execute(Tensor& a, float b)  { ufunc::scalar_inplace<DivOp>(a, b); }

Tensor AddImpl<Device::CPU>::execute(const Tensor& a, float b)  { return ufunc::scalar_out<AddOp>(a, b); }
Tensor SubImpl<Device::CPU>::execute(const Tensor& a, float b)  { return ufunc::scalar_out<SubOp>(a, b); }
Tensor MulImpl<Device::CPU>::execute(const Tensor& a, float b)  { return ufunc::scalar_out<MulOp>(a, b); }
Tensor DivImpl<Device::CPU>::execute(const Tensor& a, float b)  { return ufunc::scalar_out<DivOp>(a, b); }

Tensor AddImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) { return ufunc::binary_op<AddOp>(a, b); }
Tensor SubImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) { return ufunc::binary_op<SubOp>(a, b); }
Tensor MulImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) { return ufunc::binary_op<MulOp>(a, b); }
Tensor DivImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) { return ufunc::binary_op<DivOp>(a, b); }

void AddImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b, Tensor& dst) { ufunc::binary_op<AddOp>(a, b, dst); }
void SubImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b, Tensor& dst) { ufunc::binary_op<SubOp>(a, b, dst); }
void MulImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b, Tensor& dst) { ufunc::binary_op<MulOp>(a, b, dst); }
void DivImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b, Tensor& dst) { ufunc::binary_op<DivOp>(a, b, dst); }

Tensor AbsImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<AbsOp>(a); }
void  AbsImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<AbsOp>(a); }
Tensor SinImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<SinOp>(a); }
void  SinImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<SinOp>(a); }
Tensor CosImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<CosOp>(a); }
void  CosImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<CosOp>(a); }
Tensor TanImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<TanOp>(a); }
void  TanImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<TanOp>(a); }
Tensor ExpImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<ExpOp>(a); }
void  ExpImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<ExpOp>(a); }
Tensor SqrtImpl<Device::CPU>::execute(const Tensor& a) { return ufunc::unary_op<SqrtOp>(a); }
void  SqrtImpl<Device::CPU>::execute(Tensor& a)        { ufunc::unary_inplace<SqrtOp>(a); }
Tensor ErfImpl<Device::CPU>::execute(const Tensor& a)  { return ufunc::unary_op<ErfOp>(a); }
Tensor PowImpl<Device::CPU>::execute(const Tensor& a, float val) { return ufunc::unary_op<PowOp>(a, val); }
void  PowImpl<Device::CPU>::execute(Tensor& a, float val)        { ufunc::unary_inplace<PowOp>(a, val); }
Tensor LogImpl<Device::CPU>::execute(const Tensor& a, float val) { return ufunc::unary_op<LogOp>(a, val); }
void  LogImpl<Device::CPU>::execute(Tensor& a, float val)        { ufunc::unary_inplace<LogOp>(a, val); }
Tensor ClampImpl<Device::CPU>::execute(const Tensor& a, float min, float max) { return ufunc::unary_op<ClampOp>(a, min, max); }
void  ClampImpl<Device::CPU>::execute(Tensor& a, float min, float max)        { ufunc::unary_inplace<ClampOp>(a, min, max); }

}  // namespace ops
