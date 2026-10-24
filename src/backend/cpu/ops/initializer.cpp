#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include "_simd_math.h"
#include <print>
#include <random>
#include <vector>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"


namespace ops {

template <typename T>
inline void fill_value_impl(T* ptr, float val, size_t numel) {
    std::fill_n(static_cast<T*>(ptr), numel, T(val));
}
template <typename T>
void fill_random_impl(T* typed_ptr, size_t numel, float min, float max) {
    #pragma omp parallel
    {
        unsigned int seed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef _OPENMP
        seed += omp_get_thread_num() * 9973;
#endif
        std::mt19937 rng(seed);
        #pragma omp for
        for (size_t i = 0; i < numel; ++i) {
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(static_cast<T>(min), static_cast<T>(max));
                typed_ptr[i] = dist(rng);
            } else {
                std::uniform_real_distribution<float> dist(min, max);
                typed_ptr[i] = static_cast<T>(dist(rng));
            }
        }
    }
}

template <typename T>
void fill_randn_impl(T* typed_ptr, size_t numel, float mean, float stddev) {
    if constexpr (std::is_same_v<T, float32>) {
        // f32 快路径：AVX2 向量化 Box-Muller（z = sqrt(-2·ln(U1))·cos/sin(2π·U2)）
        float* p = static_cast<float*>(typed_ptr);
        const size_t vec_end = numel & ~size_t(15);  // 每轮 16 个输出（8 对）
        #pragma omp parallel
        {
            unsigned int seed = static_cast<unsigned int>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef _OPENMP
            seed += omp_get_thread_num() * 9973;
#endif
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> udist(0.0f, 1.0f);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < vec_end; i += 16) {
                float a1[8], a2[8];
                for (int j = 0; j < 8; ++j) { a1[j] = udist(rng); a2[j] = udist(rng); }
                __m256 u1 = _mm256_loadu_ps(a1);
                __m256 u2 = _mm256_loadu_ps(a2);
                u1 = _mm256_max_ps(u1, _mm256_set1_ps(1e-12f));  // 防 ln(0)
                const __m256 r = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), log_ps(u1)));
                const __m256 th = _mm256_mul_ps(_mm256_set1_ps(6.283185307179586f), u2);  // 2π·U2
                const __m256 vm = _mm256_set1_ps(mean);
                const __m256 vs = _mm256_set1_ps(stddev);
                _mm256_storeu_ps(p + i,     _mm256_fmadd_ps(_mm256_mul_ps(r, cos_ps(th)), vs, vm));
                _mm256_storeu_ps(p + i + 8, _mm256_fmadd_ps(_mm256_mul_ps(r, sin_ps(th)), vs, vm));
            }
        }
        // 尾（<=15 个）：精确 std::normal_distribution，单线程即可
        if (vec_end < numel) {
            unsigned int seed = static_cast<unsigned int>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::mt19937 rng(seed);
            std::normal_distribution<float> dist(mean, stddev);
            for (size_t i = vec_end; i < numel; ++i)
                p[i] = dist(rng);
        }
        return;
    }
    #pragma omp parallel
    {
        unsigned int seed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef _OPENMP
        seed += omp_get_thread_num() * 9973;
#endif
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(mean, stddev);
        #pragma omp for
        for (size_t i = 0; i < numel; ++i)
            typed_ptr[i] = static_cast<T>(dist(rng));
    }
}

//****************************** 特化 ***********************************
Tensor ZerosImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, 0,res.numel());
    });
    return res;
}

Tensor OnesImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype) {
   Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, 1,res.numel());
    });
    return res;
}

Tensor FillImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float value) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, value,res.numel());
    });
    return res;
}

void FillImpl<Device::CPU>::execute(Tensor& a, float value) {
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(a.data());
        std::fill(p, p + a.numel(), static_cast<T>(value));
    });
}

Tensor RandomImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_random_impl<T>(res_ptr,res.numel(),min,max);
    });
    return res;
}

Tensor RandnImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float mean, float std) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_randn_impl<T>(res_ptr, res.numel(), mean, std);
    });
    return res;
}

Tensor ArangeImpl<Device::CPU>::execute(float start, float stop, float step, DataType dtype) {
    // 值用 start + i*step 直接计算（不累积浮点误差），numel 由区间长度上取整
    const int64_t n = std::max<int64_t>(0, static_cast<int64_t>(std::ceil((stop - start) / step)));
    Tensor res({n}, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(res.data());
        #pragma omp parallel for
        for (int64_t i = 0; i < n; ++i)
            p[i] = static_cast<T>(start + static_cast<float>(i) * step);
    });
    return res;
}

Tensor LinspaceImpl<Device::CPU>::execute(float start, float stop, int64_t steps, DataType dtype) {
    Tensor res({steps}, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(res.data());
        if (steps == 1) { p[0] = static_cast<T>(start); return; }
        const float inv = 1.0f / static_cast<float>(steps - 1);
        #pragma omp parallel for
        for (int64_t i = 0; i < steps; ++i)
            p[i] = static_cast<T>(start + (stop - start) * (static_cast<float>(i) * inv));
    });
    return res;
}

Tensor EyeImpl<Device::CPU>::execute(int64_t n, int64_t m, DataType dtype) {
    Tensor res({n, m}, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(res.data());
        std::fill_n(p, static_cast<size_t>(n * m), T(0));
        const int64_t d = std::min(n, m);
        for (int64_t i = 0; i < d; ++i)
            p[i * m + i] = T(1);
    });
    return res;
}

}  // namespace ops
