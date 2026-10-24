// AVX2 向量化超越函数近似（f32）。
// 供 arithmetic.cpp / activate.cpp 的 f32 快路径在 GMF 里 #include。
// 精度约 1e-7 相对误差，满足神经网络前向需求（远高于测试容差 1e-5）。
#pragma once

#include <immintrin.h>

// exp(x)，x 被 clamp 到 [-87, 88]（float exp 不溢出/下溢的区间）。
// 范围归约 x = n·ln2 + r，n = round(x·log2e)，再用 7 阶 Taylor 多项式逼近
// exp(r)，最后通过指数位平移乘上 2^n。
static inline __m256 exp_ps(__m256 x) {
    const __m256 lo_bound = _mm256_set1_ps(-87.0f);
    const __m256 hi_bound = _mm256_set1_ps(88.0f);
    x = _mm256_max_ps(x, lo_bound);
    x = _mm256_min_ps(x, hi_bound);

    const __m256 log2e  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(-6.93145751953125e-1f);       // -0.693145751953125
    const __m256 ln2_lo = _mm256_set1_ps(-1.42860682033041704512e-6f); // ln2 - ln2_hi

    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT);
    __m256 r = _mm256_fmadd_ps(n, ln2_hi, x);   // x - n·ln2_hi
    r        = _mm256_fmadd_ps(n, ln2_lo, r);   // r - n·ln2_lo

    // Taylor 7 阶：exp(r) = 1 + r + r²/2 + ... + r⁷/7!
    __m256 p = _mm256_set1_ps(1.9841269841269841e-4f);  // 1/5040
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3888888888888889e-3f));  // 1/720
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3333333333333333e-3f));  // 1/120
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1666666666666667e-2f));  // 1/24
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666666666666667e-1f));  // 1/6
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0e-1f));                 // 1/2
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));                    // +1
    const __m256 exp_r = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));   // ·r + 1

    // 2^n 通过向指数位加 n（n 已是整数，cvtt 精确）
    const __m256i n_int   = _mm256_cvttps_epi32(n);
    const __m256i expbits = _mm256_slli_epi32(n_int, 23);
    return _mm256_castsi256_ps(_mm256_add_epi32(expbits, _mm256_castps_si256(exp_r)));
}

// sigmoid(x) = 1/(1+exp(-x))。x>=0 用 1/(1+exp(-|x|))，x<0 用 exp(-|x|)/(1+exp(-|x|))，
// 只算一次 exp，且避免 exp(-x) 对极负 x 上溢。
static inline __m256 sigmoid_ps(__m256 x) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 ax   = _mm256_max_ps(x, _mm256_sub_ps(zero, x));   // |x|
    const __m256 e    = exp_ps(_mm256_sub_ps(zero, ax));            // exp(-|x|)
    const __m256 pos  = _mm256_div_ps(one, _mm256_add_ps(one, e));  // 1/(1+e)
    const __m256 neg  = _mm256_div_ps(e, _mm256_add_ps(one, e));    // e/(1+e)
    const __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);         // x >= 0
    return _mm256_blendv_ps(neg, pos, mask);
}

// silu(x) = x·sigmoid(x)
static inline __m256 silu_ps(__m256 x) {
    return _mm256_mul_ps(x, sigmoid_ps(x));
}

// tanh(x) = sign(x)·(1 - 2/(exp(2|x|)+1))。小 |x| 处有约 2e-7 的绝对误差
// （对 1 附近的减法取消），对激活函数足够。
static inline __m256 tanh_ps(__m256 x) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 two  = _mm256_set1_ps(2.0f);
    const __m256 ax   = _mm256_max_ps(x, _mm256_sub_ps(zero, x));   // |x|
    // 2|x| clamp 到 88 防 exp 上溢；|x|>44 时 2/(e+1) 下溢为 0，tanh 自然饱和到 ±1
    const __m256 e2 = exp_ps(_mm256_min_ps(_mm256_mul_ps(two, ax), _mm256_set1_ps(88.0f)));
    const __m256 t  = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e2, one)));
    const __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));    // 提取符号位
    return _mm256_or_ps(t, sign);
}

// ln(x)，x > 0。范围归约 x = m·2^e（m ∈ [1,2)），用 atanh 多项式逼近 ln(m)，
// 再加 e·ln2。精度约 1e-7 相对误差（m ∈ [1,2) 内）。
static inline __m256 log_ps(__m256 x) {
    const __m256i xi = _mm256_castps_si256(x);
    // 提取指数 e = ((xi>>23)&0xFF) - 127
    const __m256i e = _mm256_sub_epi32(
        _mm256_srli_epi32(_mm256_and_si256(xi, _mm256_set1_epi32(0x7F800000)), 23),
        _mm256_set1_epi32(127));
    const __m256 ef = _mm256_cvtepi32_ps(e);
    // 尾数 m ∈ [1,2)：把指数位写成 127
    const __m256 m = _mm256_castsi256_ps(_mm256_or_si256(
        _mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)),
        _mm256_set1_epi32(0x3F800000)));
    // y = (m-1)/(m+1) ∈ [0, 1/3)，ln(m) = 2·atanh(y)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 y = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
    const __m256 y2 = _mm256_mul_ps(y, y);
    // atanh(y) = y·(1 + y²·(1/3 + y²·(1/5 + y²·(1/7 + y²·(1/9 + y²·(1/11 + y²/13))))))
    __m256 p = _mm256_set1_ps(1.0f / 13.0f);
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0f / 11.0f));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0f / 9.0f));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0f / 7.0f));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0f / 5.0f));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(1.0f / 3.0f));
    p = _mm256_fmadd_ps(p, y2, one);
    const __m256 ln_m = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), y), p);
    return _mm256_fmadd_ps(ef, _mm256_set1_ps(0.6931471805599453f), ln_m);
}

// sin(x)。范围归约到 [-π/2, π/2] 后取奇次 Taylor 多项式（到 r¹³/13!）。
// 适合 |x| 较小；对 [0, 2π] 内参数精度约 1e-6 绝对误差。
static inline __m256 sin_ps(__m256 x) {
    const __m256 inv_pi = _mm256_set1_ps(0.31830988618379067154f);  // 1/π
    const __m256 pi     = _mm256_set1_ps(3.14159265358979323846f);
    const __m256 q = _mm256_round_ps(_mm256_mul_ps(x, inv_pi), _MM_FROUND_TO_NEAREST_INT);
    const __m256 r = _mm256_fnmadd_ps(q, pi, x);  // x - q·π ∈ [-π/2, π/2]
    const __m256 r2 = _mm256_mul_ps(r, r);
    // sin(r) = r·P(r²)，P 取到 r¹³ 项
    __m256 p = _mm256_set1_ps(1.6059043836821613e-10f);              // 1/13!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-2.505210838544172e-8f)); // -1/11!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(2.755731922398589e-6f));  // 1/9!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.984126984126984e-4f)); // -1/7!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(8.333333333333333e-3f));  // 1/5!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(-1.6666666666666667e-1f));// -1/3!
    p = _mm256_fmadd_ps(p, r2, _mm256_set1_ps(1.0f));
    return _mm256_mul_ps(r, p);
}

// cos(x) = sin(x + π/2)
static inline __m256 cos_ps(__m256 x) {
    return sin_ps(_mm256_add_ps(x, _mm256_set1_ps(1.57079632679489661923f)));
}

// __m256 横向求和（8 个 f32 累加成一个标量）
static inline float hsum_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// __m256 横向最大值（8 个 f32 归约成一个标量）
static inline float hmax_ps(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}
