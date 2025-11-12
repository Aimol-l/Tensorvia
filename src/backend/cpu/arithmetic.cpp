#include "backend/cpu/ops/arithmetic.h"


namespace ops {

template <typename T>
void add_kernel(const T* RESTRICT a, const T* RESTRICT b, T* RESTRICT out, size_t size) {
    // #pragma omp parallfor
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
void sub_kernel(const T* a, const T* b, T* out, size_t size) {
    #pragma omp parallel for
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
void dot_kernel(const T* a, const T* b, T* out, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<T>(a[i] * b[i]);  // 注意类型转换
    }
}
template <typename T>
void dot_kernel(T* a, float b, size_t size) {
    const T b_cast = static_cast<T>(b); // 注意类型转换
    for (size_t i = 0; i < size; ++i) {
        a[i] *= b_cast;  // 注意类型转换
    }
}
template <typename T>
void div_kernel(const T* a, const T* b, T* out, size_t size) {
    #pragma omp parallel for
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
void abs_kernel(const T* src_ptr, T* dst_ptr, size_t n) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        dst_ptr[i] = src_ptr[i] > T(0) ? src_ptr[i] : -src_ptr[i];
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
#pragma omp parallel for
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
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            float r = std::logf(float(src_ptr[i])) / std::logf(val);
            dst_ptr[i] = static_cast<R>(r);
        } else {
            dst_ptr[i] = static_cast<R>(std::log(src_ptr[i]) / std::log(T(val)));
        }
    }
}

// ========================================================================

void AddImpl<Device::CPU>::execute(Tensor& a, float b) {
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        add_kernel<T>(static_cast<T*>(a.data()), b, a.numel());
    });
}
Tensor AddImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) return ops::Add(a.clone(), b.clone());
    // 快速路径：相同类型且无需转换
    if (a.dtype() == b.dtype()) {
        Tensor result(a.shape(), a.dtype(), Device::CPU);
        dispatch_dtype(a.dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            add_kernel<T>(
                static_cast<const T*>(a.data()),
                static_cast<const T*>(b.data()),
                static_cast<T*>(result.data()),
                a.numel()
            );
        });
        return result;
    }
    // 慢速路径：类型不同，需要 Typecast
    DataType res_type = compute_type(a.dtype(), b.dtype());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a, res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b, res_type);
    Tensor result(a.shape(), res_type, Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(A.data());
        const T* b_ptr = static_cast<const T*>(B.data());
        T* res_ptr = static_cast<T*>(result.data());
        add_kernel<T>(a_ptr,b_ptr,res_ptr,a.numel());
    });
    return result;
}
void AddImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b,Tensor& dst) {
    // 快速路径：相同类型且无需转换
    if (a.dtype() == b.dtype()) {
        dispatch_dtype(a.dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            add_kernel<T>(
                static_cast<const T*>(a.data()),
                static_cast<const T*>(b.data()),
                static_cast<T*>(dst.data()),
                a.numel()
            );
        });
        return; // ✅ 关键：快速路径后直接返回！
    }
    // 慢速路径：类型不同，需要 Typecast
    DataType res_type = compute_type(a.dtype(), b.dtype());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a, res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b, res_type);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(A.data());
        const T* b_ptr = static_cast<const T*>(B.data());
        T* res_ptr = static_cast<T*>(dst.data());
        add_kernel<T>(a_ptr,b_ptr,res_ptr,a.numel());
    });
}
Tensor AddImpl<Device::CPU>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    ops::Add(t,b);
    return t;
}

void SubImpl<Device::CPU>::execute(Tensor& a, float b) {
   dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        sub_kernel<T>(static_cast<T*>(a.data()), b, a.numel());
    });
}

Tensor SubImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    // 避免自修改
    if (&a == &b)   return ops::Sub(a.clone(), b.clone());
    // DataType res_type = std::max(a.dtype(), b.dtype());  // 全是int 或 全是 float
    // if (a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64) {
    //     res_type = std::max(b.dtype(), DataType::FLOAT32);
    // } else if (a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64) {
    //     res_type = std::max(a.dtype(), DataType::FLOAT32);
    // }
    DataType res_type = compute_type(a.dtype(), b.dtype());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a, res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b, res_type);
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::CPU);
    switch (res_type) {
        case DataType::INT8:
            sub_kernel<int8_t>(static_cast<const int8_t*>(A.data()), static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(result.data()), size);
            break;
        case DataType::INT16:
            sub_kernel<int16_t>(static_cast<const int16_t*>(A.data()), static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(result.data()), size);
            break;
        case DataType::INT32:
            sub_kernel<int32_t>(static_cast<const int32_t*>(A.data()), static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(result.data()), size);
            break;
        case DataType::INT64:
            sub_kernel<int64_t>(static_cast<const int64_t*>(A.data()), static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(result.data()), size);
            break;
        case DataType::FLOAT16:
            sub_kernel<float16>(static_cast<const float16*>(A.data()), static_cast<const float16*>(B.data()), static_cast<float16*>(result.data()), size);
            break;
        case DataType::BFLOAT16:
            sub_kernel<bfloat16>(static_cast<const bfloat16*>(A.data()), static_cast<const bfloat16*>(B.data()), static_cast<bfloat16*>(result.data()), size);
            break;
        case DataType::FLOAT32:
            sub_kernel<float32>(static_cast<const float32*>(A.data()), static_cast<const float32*>(B.data()), static_cast<float32*>(result.data()), size);
            break;
        case DataType::FLOAT64:
            sub_kernel<float64>(static_cast<const float64*>(A.data()), static_cast<const float64*>(B.data()), static_cast<float64*>(result.data()), size);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for sub");
    }
    return result;
}
Tensor SubImpl<Device::CPU>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    ops::Sub(t,b);
    return t;
}
void DotImpl<Device::CPU>::execute(Tensor& a, float b) {
   dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        dot_kernel<T>(static_cast<T*>(a.data()), b, a.numel());
    });
}
Tensor DotImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    // 避免自修改
    if (&a == &b)   return ops::Dot(a.clone(), b.clone());

    // DataType res_type = std::max(a.dtype(), b.dtype());  // 全是int 或 全是 float
    // if (a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64) {
    //     res_type = std::max(b.dtype(), DataType::FLOAT32);
    // } else if (a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64) {
    //     res_type = std::max(a.dtype(), DataType::FLOAT32);
    // }
    
    DataType res_type = compute_type(a.dtype(), b.dtype());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a, res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b, res_type);
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::CPU);
    switch (res_type) {
        case DataType::INT8:
            dot_kernel<int8_t>(static_cast<const int8_t*>(A.data()), static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(result.data()), size);
            break;
        case DataType::INT16:
            dot_kernel<int16_t>(static_cast<const int16_t*>(A.data()), static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(result.data()), size);
            break;
        case DataType::INT32:
            dot_kernel<int32_t>(static_cast<const int32_t*>(A.data()), static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(result.data()), size);
            break;
        case DataType::INT64:
            dot_kernel<int64_t>(static_cast<const int64_t*>(A.data()), static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(result.data()), size);
            break;
        case DataType::FLOAT16:
            dot_kernel<float16>(static_cast<const float16*>(A.data()), static_cast<const float16*>(B.data()), static_cast<float16*>(result.data()), size);
            break;
        case DataType::BFLOAT16:
            dot_kernel<bfloat16>(static_cast<const bfloat16*>(A.data()), static_cast<const bfloat16*>(B.data()), static_cast<bfloat16*>(result.data()), size);
            break;
        case DataType::FLOAT32:
            dot_kernel<float32>(static_cast<const float32*>(A.data()), static_cast<const float32*>(B.data()), static_cast<float32*>(result.data()), size);
            break;
        case DataType::FLOAT64:
            dot_kernel<float64>(static_cast<const float64*>(A.data()), static_cast<const float64*>(B.data()), static_cast<float64*>(result.data()), size);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for dot");
    }
    return result;
}
Tensor DotImpl<Device::CPU>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    ops::Dot(t,b);
    return t;
}
void DivImpl<Device::CPU>::execute(Tensor& a, float b) {
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        div_kernel<T>(static_cast<T*>(a.data()), b, a.numel());
    });
}
Tensor DivImpl<Device::CPU>::execute(const Tensor& a, const Tensor& b) {
    // 避免加修改
    if (&a == &b)
        return DivImpl<Device::CPU>::execute(a.clone(), b.clone());

    const Tensor& A = a.dtype() > b.dtype() ? a : ops::Typecast(a, b.dtype());
    const Tensor& B = a.dtype() <= b.dtype() ? b : ops::Typecast(b, a.dtype());

    size_t size = A.numel();
    Tensor result(A.shape(), A.dtype(), Device::CPU);
    // 分发到模板 kernel（根据 dtype 决定类型）
    switch (A.dtype()) {
        case DataType::INT8:
            div_kernel<int8_t>(static_cast<const int8_t*>(A.data()), static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(result.data()), size);
            break;
        case DataType::INT16:
            div_kernel<int16_t>(static_cast<const int16_t*>(A.data()), static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(result.data()), size);
            break;
        case DataType::INT32:
            div_kernel<int32_t>(static_cast<const int32_t*>(A.data()), static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(result.data()), size);
            break;
        case DataType::INT64:
            div_kernel<int64_t>(static_cast<const int64_t*>(A.data()), static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(result.data()), size);
            break;
        case DataType::FLOAT16:
            div_kernel<float16>(static_cast<const float16*>(A.data()), static_cast<const float16*>(B.data()), static_cast<float16*>(result.data()), size);
            break;
        case DataType::BFLOAT16:
            div_kernel<bfloat16>(static_cast<const bfloat16*>(A.data()), static_cast<const bfloat16*>(B.data()), static_cast<bfloat16*>(result.data()), size);
            break;
        case DataType::FLOAT32:
            div_kernel<float32>(static_cast<const float32*>(A.data()), static_cast<const float32*>(B.data()), static_cast<float32*>(result.data()), size);
            break;
        case DataType::FLOAT64:
            div_kernel<float64>(static_cast<const float64*>(A.data()), static_cast<const float64*>(B.data()), static_cast<float64*>(result.data()), size);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for div");
    }
    return result;
}
Tensor DivImpl<Device::CPU>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    ops::Div(t,b);
    return t;
}
void AbsImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        case DataType::INT8:
            abs_kernel<int8_t>(static_cast<const int8_t*>(src), static_cast<int8_t*>(dst), a.numel());
            break;
        case DataType::INT16:
            abs_kernel<int16_t>(static_cast<const int16_t*>(src), static_cast<int16_t*>(dst), a.numel());
            break;
        case DataType::INT32:
            abs_kernel<int32_t>(static_cast<const int32_t*>(src), static_cast<int32_t*>(dst), a.numel());
            break;
        case DataType::INT64:
            abs_kernel<int64_t>(static_cast<const int64_t*>(src), static_cast<int64_t*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            abs_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            abs_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        case DataType::FLOAT32:
            abs_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            abs_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor AbsImpl<Device::CPU>::execute(const Tensor& a) {
    Tensor b = a.clone();
    ops::Abs(b);
    return b;
}
void SinImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        sin_kernel<int8_t, float32>(static_cast<int8_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT16:       sin_kernel<int16_t, float32>(static_cast<int16_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT32:       sin_kernel<int32_t, float32>(static_cast<int32_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT64:       sin_kernel<int64_t, float32>(static_cast<int64_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        case DataType::FLOAT32:
            sin_kernel<float32>(static_cast<float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            sin_kernel<float64>(static_cast<float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            sin_kernel<float16>(static_cast<float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            sin_kernel<bfloat16>(static_cast<bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor SinImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            sin_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT16:
            sin_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT32:
            sin_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT64:
            sin_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT32:
            sin_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT64:
            sin_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel());
            break;
        case DataType::FLOAT16:
            sin_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel());
            break;
        case DataType::BFLOAT16:
            sin_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void CosImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        cos_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT16:       cos_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT32:       cos_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT64:       cos_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        case DataType::FLOAT32:
            cos_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            cos_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            cos_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            cos_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor CosImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            cos_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT16:
            cos_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT32:
            cos_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT64:
            cos_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT32:
            cos_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT64:
            cos_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel());
            break;
        case DataType::FLOAT16:
            cos_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel());
            break;
        case DataType::BFLOAT16:
            cos_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void TanImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        tan_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT16:       tan_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT32:       tan_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT64:       tan_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        case DataType::FLOAT32:
            tan_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            tan_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            tan_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            tan_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor TanImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            tan_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT16:
            tan_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT32:
            tan_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT64:
            tan_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT32:
            tan_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT64:
            tan_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel());
            break;
        case DataType::FLOAT16:
            tan_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel());
            break;
        case DataType::BFLOAT16:
            tan_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void ExpImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        exp_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT16:       exp_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT32:       exp_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT64:       exp_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        case DataType::FLOAT32:
            exp_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            exp_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            exp_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            exp_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor ExpImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            exp_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT16:
            exp_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT32:
            exp_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT64:
            exp_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT32:
            exp_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT64:
            exp_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel());
            break;
        case DataType::FLOAT16:
            exp_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel());
            break;
        case DataType::BFLOAT16:
            exp_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void SqrtImpl<Device::CPU>::execute(Tensor& a) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        sqrt_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT16:       sqrt_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT32:       sqrt_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        // case DataType::INT64:       sqrt_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel()); break;
        case DataType::FLOAT32:
            sqrt_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel());
            break;
        case DataType::FLOAT64:
            sqrt_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel());
            break;
        case DataType::FLOAT16:
            sqrt_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel());
            break;
        case DataType::BFLOAT16:
            sqrt_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor SqrtImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            sqrt_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT16:
            sqrt_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT32:
            sqrt_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::INT64:
            sqrt_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT32:
            sqrt_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel());
            break;
        case DataType::FLOAT64:
            sqrt_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel());
            break;
        case DataType::FLOAT16:
            sqrt_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel());
            break;
        case DataType::BFLOAT16:
            sqrt_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel());
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void LogImpl<Device::CPU>::execute(Tensor& a, float val) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        // case DataType::INT8:        log_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel(),val); break;
        // case DataType::INT16:       log_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel(),val); break;
        // case DataType::INT32:       log_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel(),val); break;
        // case DataType::INT64:       log_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel(),val); break;
        case DataType::FLOAT32:
            log_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::FLOAT64:
            log_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel(), val);
            break;
        case DataType::FLOAT16:
            log_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel(), val);
            break;
        case DataType::BFLOAT16:
            log_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel(), val);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor LogImpl<Device::CPU>::execute(const Tensor& a, float val) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            log_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT16:
            log_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT32:
            log_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT64:
            log_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT32:
            log_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT64:
            log_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT16:
            log_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel(), val);
            break;
        case DataType::BFLOAT16:
            log_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel(), val);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void PowImpl<Device::CPU>::execute(Tensor& a, float val) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        case DataType::INT8:
            pow_kernel<int8_t, float32>(static_cast<const int8_t*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::INT16:
            pow_kernel<int16_t, float32>(static_cast<const int16_t*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::INT32:
            pow_kernel<int32_t, float32>(static_cast<const int32_t*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::INT64:
            pow_kernel<int64_t, float32>(static_cast<const int64_t*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::FLOAT32:
            pow_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel(), val);
            break;
        case DataType::FLOAT64:
            pow_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel(), val);
            break;
        case DataType::FLOAT16:
            pow_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel(), val);
            break;
        case DataType::BFLOAT16:
            pow_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel(), val);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor PowImpl<Device::CPU>::execute(const Tensor& a, float val) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor b = Tensor(a.shape(), res_type, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:
            pow_kernel<int8_t, float32>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT16:
            pow_kernel<int16_t, float32>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT32:
            pow_kernel<int32_t, float32>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::INT64:
            pow_kernel<int64_t, float32>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT32:
            pow_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT64:
            pow_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(b.data()), a.numel(), val);
            break;
        case DataType::FLOAT16:
            pow_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(b.data()), a.numel(), val);
            break;
        case DataType::BFLOAT16:
            pow_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(b.data()), a.numel(), val);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return b;
}
void ClampImpl<Device::CPU>::execute(Tensor& a, float min, float max) {
    void* src = a.data();
    void* dst = a.data();
    switch (a.dtype()) {
        case DataType::INT8:
            clamp_kernel<int8_t>(static_cast<const int8_t*>(src), static_cast<int8_t*>(dst), a.numel(), min, max);
            break;
        case DataType::INT16:
            clamp_kernel<int16_t>(static_cast<const int16_t*>(src), static_cast<int16_t*>(dst), a.numel(), min, max);
            break;
        case DataType::INT32:
            clamp_kernel<int32_t>(static_cast<const int32_t*>(src), static_cast<int32_t*>(dst), a.numel(), min, max);
            break;
        case DataType::INT64:
            clamp_kernel<int64_t>(static_cast<const int64_t*>(src), static_cast<int64_t*>(dst), a.numel(), min, max);
            break;
        case DataType::BFLOAT16:
            clamp_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(dst), a.numel(), min, max);
            break;
        case DataType::FLOAT16:
            clamp_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(dst), a.numel(), min, max);
            break;
        case DataType::FLOAT32:
            clamp_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(dst), a.numel(), min, max);
            break;
        case DataType::FLOAT64:
            clamp_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(dst), a.numel(), min, max);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
}
Tensor ClampImpl<Device::CPU>::execute(const Tensor& a, float min, float max) {
    const void* src = a.data();
    Tensor res = Tensor(a.shape(), a.dtype(), Device::CPU);
    switch (a.dtype()) {
        case DataType::FLOAT64:
            clamp_kernel<float64>(static_cast<const float64*>(src), static_cast<float64*>(res.data()), a.numel(), min, max);
            break;
        case DataType::FLOAT32:
            clamp_kernel<float32>(static_cast<const float32*>(src), static_cast<float32*>(res.data()), a.numel(), min, max);
            break;
        case DataType::FLOAT16:
            clamp_kernel<float16>(static_cast<const float16*>(src), static_cast<float16*>(res.data()), a.numel(), min, max);
            break;
        case DataType::BFLOAT16:
            clamp_kernel<bfloat16>(static_cast<const bfloat16*>(src), static_cast<bfloat16*>(res.data()), a.numel(), min, max);
            break;
        case DataType::INT64:
            clamp_kernel<int64_t>(static_cast<const int64_t*>(src), static_cast<int64_t*>(res.data()), a.numel(), min, max);
            break;
        case DataType::INT32:
            clamp_kernel<int32_t>(static_cast<const int32_t*>(src), static_cast<int32_t*>(res.data()), a.numel(), min, max);
            break;
        case DataType::INT16:
            clamp_kernel<int16_t>(static_cast<const int16_t*>(src), static_cast<int16_t*>(res.data()), a.numel(), min, max);
            break;
        case DataType::INT8:
            clamp_kernel<int8_t>(static_cast<const int8_t*>(src), static_cast<int8_t*>(res.data()), a.numel(), min, max);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return res;
}

template struct AddImpl<Device::CPU>;
template struct SubImpl<Device::CPU>;
template struct DotImpl<Device::CPU>;
template struct DivImpl<Device::CPU>;
template struct SinImpl<Device::CPU>;
template struct CosImpl<Device::CPU>;
template struct TanImpl<Device::CPU>;
template struct PowImpl<Device::CPU>;
template struct LogImpl<Device::CPU>;
template struct ExpImpl<Device::CPU>;
template struct SqrtImpl<Device::CPU>;
template struct AbsImpl<Device::CPU>;
template struct ClampImpl<Device::CPU>;
}