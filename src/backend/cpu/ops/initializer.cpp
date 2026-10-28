#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "_simd_math.h"
#include <print>
#include <random>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"


namespace ops {

template <typename T>
inline void fill_value_impl(T* ptr, float val, size_t numel) {
    const T v = static_cast<T>(val);
    #pragma omp parallel for if (numel > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < numel; ++i) ptr[i] = v;
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

// ---------------------------------------------------------------------------
// f32 randn 的向量化 RNG。
//
// 原来的实现用 std::mt19937 + uniform_real_distribution 逐元素抽两个 [0,1)
// 样本：16M 个输出要 32M 次标量抽取，AVX2 只加速了后面的 ln/cos/sin，RNG 才是
// 瓶颈。下面把 xorshift128 的 4 个状态字各放一个 __m256i，8 个通道各自独立
// 进展，每一步产出 8 个均匀样本；种子用 splitmix64 展开到各通道，互不相关。
// ---------------------------------------------------------------------------
inline __m256i make_rng_lane(std::uint64_t base) {
    alignas(32) std::uint32_t lanes[8];
    for (int lane = 0; lane < 8; ++lane) {
        std::uint64_t z = base + static_cast<std::uint64_t>(lane) * 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        // |1：xorshift 状态不能全零
        lanes[lane] = static_cast<std::uint32_t>((z ^ (z >> 31)) >> 32) | 1u;
    }
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(lanes));
}

inline __m256i xorshift128_next(__m256i& x, __m256i& y, __m256i& z, __m256i& w) {
    const __m256i t = _mm256_xor_si256(x, _mm256_slli_epi32(x, 11));
    x = y;
    y = z;
    z = w;
    w = _mm256_xor_si256(_mm256_xor_si256(w, _mm256_srli_epi32(w, 19)),
                         _mm256_xor_si256(t, _mm256_srli_epi32(t, 8)));
    return w;
}

// 取高 24 位映射到 [0,1)：24 位在 float 里精确可表示，乘 2^-24 不再引入舍入。
inline __m256 to_uniform_01(__m256i state) {
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(state, 8)),
                         _mm256_set1_ps(1.0f / 16777216.0f));
}

// sin/cos 共享一次范围归约。θ = 2π·U ∈ [0, 2π)，q = round(θ/π) ∈ {0,1,2}，
// r = θ - qπ ∈ [-π/2, π/2]，sin(θ) = (-1)^q·sin(r)，cos(θ) = (-1)^q·cos(r)。
// 比 cos_ps(θ) = sin_ps(θ+π/2) 少一次完整归约和一次多项式求值。
inline void sincos_ps(__m256 x, __m256& s, __m256& c) {
    const __m256 q  = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.31830988618379067154f)),
                                      _MM_FROUND_TO_NEAREST_INT);
    const __m256 r  = _mm256_fnmadd_ps(q, _mm256_set1_ps(3.14159265358979323846f), x);
    const __m256 r2 = _mm256_mul_ps(r, r);

    __m256 ps = _mm256_set1_ps(1.6059043836821613e-10f);                 // 1/13!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(-2.505210838544172e-8f));  // -1/11!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(2.755731922398589e-6f));   // 1/9!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(-1.984126984126984e-4f));  // -1/7!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(8.333333333333333e-3f));   // 1/5!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(-1.6666666666666667e-1f)); // -1/3!
    ps = _mm256_fmadd_ps(ps, r2, _mm256_set1_ps(1.0f));

    __m256 pc = _mm256_set1_ps(2.08767569878681e-9f);                    // 1/12!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(-2.755731922398589e-7f));  // -1/10!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(2.48015873015873e-5f));    // 1/8!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(-1.388888888888889e-3f));  // -1/6!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(4.1666666666666667e-2f));  // 1/4!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(-5.0e-1f));                // -1/2!
    pc = _mm256_fmadd_ps(pc, r2, _mm256_set1_ps(1.0f));

    // q 为奇数时同时翻转 sin/cos 的符号
    const __m256i odd = _mm256_slli_epi32(
        _mm256_and_si256(_mm256_cvtps_epi32(q), _mm256_set1_epi32(1)), 31);
    const __m256 sign = _mm256_castsi256_ps(odd);
    s = _mm256_xor_ps(_mm256_mul_ps(r, ps), sign);
    c = _mm256_xor_ps(pc, sign);
}

template <typename T>
void fill_randn_impl(T* typed_ptr, size_t numel, float mean, float stddev) {
    if constexpr (std::is_same_v<T, float32>) {
        // f32 快路径：AVX2 向量化 Box-Muller（z = sqrt(-2·ln(U1))·cos/sin(2π·U2)）
        float* p = static_cast<float*>(typed_ptr);
        const size_t vec_end = numel & ~size_t(15);  // 每轮 16 个输出（8 对）
        #pragma omp parallel
        {
            std::uint64_t base = static_cast<std::uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
#ifdef _OPENMP
            base += static_cast<std::uint64_t>(omp_get_thread_num()) * 0x9E3779B97F4A7C15ull;
#endif
            __m256i x = make_rng_lane(base ^ 0xA0761D6478BD642Full);
            __m256i y = make_rng_lane(base ^ 0xE7037ED1A0B428DBull);
            __m256i z = make_rng_lane(base ^ 0x8EBC6AF09C88C6E3ull);
            __m256i w = make_rng_lane(base ^ 0x589965CC75374CC3ull);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < vec_end; i += 16) {
                const __m256 u1 = _mm256_max_ps(
                    to_uniform_01(xorshift128_next(x, y, z, w)),
                    _mm256_set1_ps(1e-12f));  // 防 ln(0)
                const __m256 u2 = to_uniform_01(xorshift128_next(x, y, z, w));
                const __m256 r = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), log_ps(u1)));
                const __m256 th = _mm256_mul_ps(_mm256_set1_ps(6.283185307179586f), u2);  // 2π·U2
                __m256 s, c;
                sincos_ps(th, s, c);
                const __m256 vm = _mm256_set1_ps(mean);
                const __m256 vs = _mm256_set1_ps(stddev);
                _mm256_storeu_ps(p + i,     _mm256_fmadd_ps(_mm256_mul_ps(r, c), vs, vm));
                _mm256_storeu_ps(p + i + 8, _mm256_fmadd_ps(_mm256_mul_ps(r, s), vs, vm));
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
        const T v = static_cast<T>(value);
        const size_t numel = a.numel();
        #pragma omp parallel for if (numel > 4096)  // 4096 是一个经验值
        for (size_t i = 0; i < numel; ++i) p[i] = v;
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
        const size_t numel = static_cast<size_t>(n * m);
        #pragma omp parallel for if (numel > 4096)  // 4096 是一个经验值
        for (size_t i = 0; i < numel; ++i) p[i] = T(0);
        const int64_t d = std::min(n, m);
        for (int64_t i = 0; i < d; ++i)
            p[i * m + i] = T(1);
    });
    return res;
}

}  // namespace ops
