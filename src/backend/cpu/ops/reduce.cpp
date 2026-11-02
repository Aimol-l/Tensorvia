#include <print>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <immintrin.h>
#include "_simd_math.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"



namespace ops {
// 索引结果固定 INT32（与 ArgMaxImpl 创建的结果张量一致），
// 不能复用输入 dtype T*：float 输入会把索引按浮点 bit pattern 写进 int32
// buffer，int8/int16 输入则直接截断溢出。
template <typename T>
void argmax_kernel(const T *src_ptr, int32_t *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);

    if constexpr (std::is_same_v<T, float32>) {
        if (n_inner == 1 && n_axis >= 8) {
            // f32 连续轴：AVX2 逐 lane 归约，横向合并时值相同取更小索引（首个最大）
            #pragma omp parallel for schedule(static)
            for (size_t outer = 0; outer < n_outer; ++outer) {
                const float* pf = src_ptr + outer * n_axis;
                __m256 vmax = _mm256_loadu_ps(pf);
                __m256i vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i lane = vidx;
                size_t i = 8;
                for (; i + 8 <= n_axis; i += 8) {
                    const __m256 cur = _mm256_loadu_ps(pf + i);
                    const __m256 gt = _mm256_cmp_ps(cur, vmax, _CMP_GT_OQ);
                    const __m256i new_idx = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(i)), lane);
                    vidx = _mm256_blendv_epi8(vidx, new_idx, _mm256_castps_si256(gt));
                    vmax = _mm256_max_ps(vmax, cur);
                }
                float vals[8];
                int32_t idxs[8];
                _mm256_storeu_ps(vals, vmax);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(idxs), vidx);
                float best_val = vals[0];
                int32_t best_idx = idxs[0];
                for (int l = 1; l < 8; ++l)
                    if (vals[l] > best_val || (vals[l] == best_val && idxs[l] < best_idx)) {
                        best_val = vals[l];
                        best_idx = idxs[l];
                    }
                for (; i < n_axis; ++i)
                    if (pf[i] > best_val) { best_val = pf[i]; best_idx = static_cast<int32_t>(i); }
                res_ptr[outer] = best_idx;
            }
            return;
        }
    }

    // 通用标量路径（非 f32 或跨步轴 / 短轴）
    const size_t total_tasks = n_outer * n_inner;
    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        const size_t outer = task_id / n_inner;
        const size_t inner = task_id % n_inner;
        const size_t start = outer * n_axis * n_inner + inner;
        T max_val = src_ptr[start];
        int32_t max_index = 0;
        for (size_t k = 1; k < n_axis; ++k) {
            const size_t pos = start + k * n_inner;
            if (src_ptr[pos] > max_val) {
                max_val = src_ptr[pos];
                max_index = static_cast<int32_t>(k);
            }
        }
        res_ptr[task_id] = max_index;
    }
}

template <typename T>
void argmin_kernel(const T *src_ptr, int32_t *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);

    if constexpr (std::is_same_v<T, float32>) {
        if (n_inner == 1 && n_axis >= 8) {
            // f32 连续轴：AVX2 逐 lane 归约，横向合并时值相同取更小索引（首个最小）
            #pragma omp parallel for schedule(static)
            for (size_t outer = 0; outer < n_outer; ++outer) {
                const float* pf = src_ptr + outer * n_axis;
                __m256 vmin = _mm256_loadu_ps(pf);
                __m256i vidx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i lane = vidx;
                size_t i = 8;
                for (; i + 8 <= n_axis; i += 8) {
                    const __m256 cur = _mm256_loadu_ps(pf + i);
                    const __m256 lt = _mm256_cmp_ps(cur, vmin, _CMP_LT_OQ);
                    const __m256i new_idx = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(i)), lane);
                    vidx = _mm256_blendv_epi8(vidx, new_idx, _mm256_castps_si256(lt));
                    vmin = _mm256_min_ps(vmin, cur);
                }
                float vals[8];
                int32_t idxs[8];
                _mm256_storeu_ps(vals, vmin);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(idxs), vidx);
                float best_val = vals[0];
                int32_t best_idx = idxs[0];
                for (int l = 1; l < 8; ++l)
                    if (vals[l] < best_val || (vals[l] == best_val && idxs[l] < best_idx)) {
                        best_val = vals[l];
                        best_idx = idxs[l];
                    }
                for (; i < n_axis; ++i)
                    if (pf[i] < best_val) { best_val = pf[i]; best_idx = static_cast<int32_t>(i); }
                res_ptr[outer] = best_idx;
            }
            return;
        }
    }

    // 通用标量路径（非 f32 或跨步轴 / 短轴）
    const size_t total_tasks = n_outer * n_inner;
    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        const size_t outer = task_id / n_inner;
        const size_t inner = task_id % n_inner;
        const size_t start = outer * n_axis * n_inner + inner;
        T min_val = src_ptr[start];
        int32_t min_index = 0;
        for (size_t k = 1; k < n_axis; ++k) {
            const size_t pos = start + k * n_inner;
            if (src_ptr[pos] < min_val) {
                min_val = src_ptr[pos];
                min_index = static_cast<int32_t>(k);
            }
        }
        res_ptr[task_id] = min_index;
    }
}

template <typename T>
float sum_kernel(const T *ptr, size_t size) {
    float sum = 0.0f;
    #pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        sum += static_cast<float>(ptr[i]);
    }
    return sum;
}

template <typename T>
float mean_kernel(const Tensor& src) {
    float total = sum_kernel<T>(static_cast<const T*>(src.data()), src.numel());  // 复用 sum_kernel
    return total / static_cast<float>(src.numel());
}

template <typename T>
float min_kernel(const T *ptr, size_t size) {
    float min_val = std::numeric_limits<float>::max();
    #pragma omp parallel for simd reduction(min:min_val) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(ptr[i]);
        min_val = std::min(min_val, val);
    }
    return min_val;
}

template <typename T>
float max_kernel(const T *ptr, size_t size) {
    float max_val = std::numeric_limits<float>::lowest();
    #pragma omp parallel for simd reduction(max:max_val) schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(ptr[i]);
        max_val = std::max(max_val, val);
    }
    return max_val;
}

// 累加类型：整型输入在 int64 里精确累加，浮点输入用 double；float32 的连续轴
// 仍走 float 的 SIMD 快路径。
template <typename T>
using reduce_acc_t = std::conditional_t<std::is_integral_v<T>, int64_t, double>;

// 输出类型 R 由 reduction_type 决定（PyTorch 语义：整型归约提升到 int64）。
template <typename T, typename R>
void sum_kernel(const T *src_ptr, R *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    using Acc = reduce_acc_t<T>;

    if (n_inner == 1) {
        // 连续轴：每个 outer 对应一段 axis_size 个连续元素
        #pragma omp parallel for schedule(static)
        for (size_t outer = 0; outer < n_outer; ++outer) {
            const T* p = src_ptr + outer * n_axis;
            Acc s = 0;
            if constexpr (std::is_same_v<T, float32>) {
                const float* pf = p;  // T == float（T 就是 float）
                __m256 acc = _mm256_setzero_ps();
                size_t i = 0;
                for (; i + 8 <= n_axis; i += 8)
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(pf + i));
                s = static_cast<Acc>(hsum_ps(acc));
                for (; i < n_axis; ++i) s += static_cast<Acc>(pf[i]);
            } else {
                #pragma omp simd reduction(+:s)
                for (size_t i = 0; i < n_axis; ++i)
                    s += static_cast<Acc>(p[i]);
            }
            res_ptr[outer] = static_cast<R>(s);
        }
    } else {
        // 跨步轴：每个 (outer, inner) 是一个归约组
        const size_t total_tasks = n_outer * n_inner;
        #pragma omp parallel for schedule(static)
        for (size_t task = 0; task < total_tasks; ++task) {
            const size_t outer = task / n_inner;
            const size_t inner = task % n_inner;
            const T* p = src_ptr + outer * n_axis * n_inner + inner;
            Acc s = 0;
            #pragma omp simd reduction(+:s)
            for (size_t k = 0; k < n_axis; ++k)
                s += static_cast<Acc>(p[k * n_inner]);
            res_ptr[task] = static_cast<R>(s);
        }
    }
}

// 输出类型 R 由 float_reduction_type 决定：整型与窄浮点归约到 float32，
// float64 输入保持双精度累加。
template <typename T, typename R>
void mean_kernel(const T *src_ptr, R *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    using Acc = std::conditional_t<std::is_same_v<R, float64>, double, float>;
    const Acc inv_axis = Acc{1} / static_cast<Acc>(n_axis);

    if (n_inner == 1) {
        // 连续轴：每个 outer 对应一段 axis_size 个连续元素
        #pragma omp parallel for schedule(static)
        for (size_t outer = 0; outer < n_outer; ++outer) {
            const T* p = src_ptr + outer * n_axis;
            Acc s = 0;
            if constexpr (std::is_same_v<T, float32> && std::is_same_v<Acc, float>) {
                const float* pf = p;  // T == float（T 就是 float）
                __m256 acc = _mm256_setzero_ps();
                size_t i = 0;
                for (; i + 8 <= n_axis; i += 8)
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(pf + i));
                s = hsum_ps(acc);
                for (; i < n_axis; ++i) s += pf[i];
            } else {
                #pragma omp simd reduction(+:s)
                for (size_t i = 0; i < n_axis; ++i)
                    s += static_cast<Acc>(p[i]);
            }
            res_ptr[outer] = static_cast<R>(s * inv_axis);
        }
    } else {
        // 跨步轴：每个 (outer, inner) 是一个归约组
        const size_t total_tasks = n_outer * n_inner;
        #pragma omp parallel for schedule(static)
        for (size_t task = 0; task < total_tasks; ++task) {
            const size_t outer = task / n_inner;
            const size_t inner = task % n_inner;
            const T* p = src_ptr + outer * n_axis * n_inner + inner;
            Acc s = 0;
            #pragma omp simd reduction(+:s)
            for (size_t k = 0; k < n_axis; ++k)
                s += static_cast<Acc>(p[k * n_inner]);
            res_ptr[task] = static_cast<R>(s * inv_axis);
        }
    }
}

template <typename T>
void min_kernel(const T *src_ptr, T *res_ptr, size_t outer_size, size_t inner_size, size_t axis_size){

    const size_t total_tasks = outer_size * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T min_val = src_ptr[start];
        // 在axis_size维度上寻找最小值索引
        if constexpr (std::is_arithmetic_v<T>) {
            // 算术类型：使用 OpenMP SIMD 向量化优化
            #pragma omp simd reduction(min:min_val)
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                min_val = std::min(min_val, src_ptr[pos]);
            }
        } else {
            // 非算术类型：使用标准循环
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                min_val = std::min(min_val, src_ptr[pos]);
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = min_val;
    }
}

template <typename T>
void max_kernel(const T *src_ptr, T *res_ptr, size_t outer_size, size_t inner_size, size_t axis_size){

    const size_t total_tasks = outer_size * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;

        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T max_val = src_ptr[start];

        // 在axis_size维度上寻找最小值索引
        if constexpr (std::is_arithmetic_v<T>) {
            #pragma omp simd reduction(max:max_val)
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                max_val = std::max(max_val, src_ptr[pos]);
            }
        }else{
                // 非算术类型：使用标准循环
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                max_val = std::max(max_val, src_ptr[pos]);
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = max_val;
    }
}

template <typename T>
float prod_kernel(const T* ptr, size_t size) {
    float prod = 1.0f;
    #pragma omp parallel for simd reduction(*:prod) schedule(static)
    for (size_t i = 0; i < size; ++i)
        prod *= static_cast<float>(ptr[i]);
    return prod;
}

template <typename T, typename R>
void prod_kernel(const T* src_ptr, R* res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    using Acc = reduce_acc_t<T>;

    if (n_inner == 1) {
        #pragma omp parallel for schedule(static)
        for (size_t outer = 0; outer < n_outer; ++outer) {
            const T* p = src_ptr + outer * n_axis;
            Acc prod = 1;
            #pragma omp simd reduction(*:prod)
            for (size_t k = 0; k < n_axis; ++k) prod *= static_cast<Acc>(p[k]);
            res_ptr[outer] = static_cast<R>(prod);
        }
    } else {
        const size_t total = n_outer * n_inner;
        #pragma omp parallel for schedule(static)
        for (size_t task = 0; task < total; ++task) {
            const size_t outer = task / n_inner;
            const size_t inner = task % n_inner;
            const T* p = src_ptr + outer * n_axis * n_inner + inner;
            Acc prod = 1;
            #pragma omp simd reduction(*:prod)
            for (size_t k = 0; k < n_axis; ++k) prod *= static_cast<Acc>(p[k * n_inner]);
            res_ptr[task] = static_cast<R>(prod);
        }
    }
}

// 总体（ddof=0）方差 / 标准差：两遍求和，先算均值再算离差平方和，避免 E[x²]-E[x]² 的相消。
template <typename T, bool Std>
float var_std_full(const T* src, size_t n) {
    const float mean = sum_kernel<T>(src, n) / static_cast<float>(n);
    float ss = 0.0f;
    #pragma omp parallel for simd reduction(+:ss) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float d = static_cast<float>(src[i]) - mean;
        ss += d * d;
    }
    const float v = ss / static_cast<float>(n);
    return Std ? std::sqrt(v) : v;
}

template <typename T, typename R, bool Std>
void var_std_kernel(const T* src_ptr, R* res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    using Acc = std::conditional_t<std::is_same_v<R, float64>, double, float>;
    const Acc inv = Acc{1} / static_cast<Acc>(n_axis);
    const size_t total = n_outer * n_inner;
    #pragma omp parallel for schedule(static)
    for (size_t task = 0; task < total; ++task) {
        const size_t outer = task / n_inner;
        const size_t inner = task % n_inner;
        const T* p = src_ptr + outer * n_axis * n_inner + inner;
        Acc sum = 0;
        for (size_t k = 0; k < n_axis; ++k) sum += static_cast<Acc>(p[k * n_inner]);
        const Acc mean = sum * inv;
        Acc ss = 0;
        for (size_t k = 0; k < n_axis; ++k) {
            const Acc d = static_cast<Acc>(p[k * n_inner]) - mean;
            ss += d * d;
        }
        const Acc v = ss * inv;
        res_ptr[task] = static_cast<R>(Std ? std::sqrt(v) : v);
    }
}

template <typename T>
float norm_full(const T* src, size_t n, float p) {
    float acc = 0.0f;
    if (p == 1.0f) {
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (size_t i = 0; i < n; ++i) acc += std::abs(static_cast<float>(src[i]));
    } else {
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const float v = static_cast<float>(src[i]);
            acc += v * v;
        }
        acc = std::sqrt(acc);
    }
    return acc;
}

template <typename T, typename R>
void norm_kernel(const T* src_ptr, R* res_ptr, int outer_size, int inner_size, int axis_size, float p) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    const bool l1 = (p == 1.0f);
    using Acc = std::conditional_t<std::is_same_v<R, float64>, double, float>;
    const size_t total = n_outer * n_inner;
    #pragma omp parallel for schedule(static)
    for (size_t task = 0; task < total; ++task) {
        const size_t outer = task / n_inner;
        const size_t inner = task % n_inner;
        const T* q = src_ptr + outer * n_axis * n_inner + inner;
        Acc acc = 0;
        if (l1) {
            for (size_t k = 0; k < n_axis; ++k) acc += std::abs(static_cast<Acc>(q[k * n_inner]));
        } else {
            for (size_t k = 0; k < n_axis; ++k) {
                const Acc v = static_cast<Acc>(q[k * n_inner]);
                acc += v * v;
            }
            acc = std::sqrt(acc);
        }
        res_ptr[task] = static_cast<R>(acc);
    }
}

template <typename T>
void cumsum_kernel(const T* src_ptr, T* dst_ptr, int outer_size, int axis_size, int inner_size) {
    const size_t n_outer = static_cast<size_t>(outer_size);
    const size_t n_inner = static_cast<size_t>(inner_size);
    const size_t n_axis  = static_cast<size_t>(axis_size);
    const size_t total = n_outer * n_inner;
    #pragma omp parallel for schedule(static)
    for (size_t task = 0; task < total; ++task) {
        const size_t outer = task / n_inner;
        const size_t inner = task % n_inner;
        const T* p = src_ptr + outer * n_axis * n_inner + inner;
        T* q = dst_ptr + outer * n_axis * n_inner + inner;
        T acc = T(0);
        for (size_t k = 0; k < n_axis; ++k) {
            acc += p[k * n_inner];
            q[k * n_inner] = acc;
        }
    }
}

template <typename T>
bool all_kernel(const T *ptr, float val, size_t n) {
    if constexpr (std::is_integral_v<T>) {
        // 整数类型：精确比较
        bool res = true;
        for(size_t i = 0; i < n; ++i){
            if(ptr[i] != static_cast<T>(val)){
                res = false;
                break;
            }
        }
        return res;
    } else {
        // 浮点类型：允许误差的比较
        const T tolerance = static_cast<T>(1e-5); // 根据需求调整容差
        bool res = true;
        for(size_t i = 0; i < n; ++i){
            if(std::abs(ptr[i] - static_cast<T>(val)) > tolerance){
                res = false;
                break;
            }
        }
        return  res;
    }
}
template <typename T>
bool any_kernel(const T *ptr, float val, size_t n) {

    constexpr auto tolerance = []{
        if constexpr (std::is_same_v<T, float>) return 1e-5f;
        else if constexpr (std::is_same_v<T, float64>) return 1e-10;
        else if constexpr (std::is_same_v<T, float16>) return 1e-3f;
        else if constexpr (std::is_same_v<T, bfloat16>) return 1e-2f;
        else return T{0}; // 整数类型
    }();

    if constexpr (std::is_integral_v<T>) {
        // 整数类型：精确比较
        bool res = false;
        for(size_t i = 0; i < n; ++i){
            if(ptr[i] == static_cast<T>(val)){
                res = true;
                break;
            }
        }
        return res;
    } else {

        // 浮点类型：允许误差的比较
        bool res = false;
        for(size_t i = 0; i < n; ++i){
            if(std::abs(ptr[i] - static_cast<T>(val)) <= tolerance){
                res = true;
                break;
            }
        }
        return res;
    }
}

// ================================================================
float SumImpl<Device::CPU>::execute(const Tensor& a) {
    float sum = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        sum = sum_kernel<T>(a_ptr,a.numel());
    });
    return sum;
}
float MinImpl<Device::CPU>::execute(const Tensor& a) {
    float min = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        min = min_kernel<T>(a_ptr,a.numel());
    });
    return min;
}
float MaxImpl<Device::CPU>::execute(const Tensor& a) {
    float max = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        max = max_kernel<T>(a_ptr,a.numel());
    });
    return max;
}
float MeanImpl<Device::CPU>::execute(const Tensor& a) {
    float mean = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        mean = mean_kernel<T>(a);
    });
    return mean;
}
// ================================================================
// 跨步单轴归约。
//
// 被归约轴在内存里连续（stride 1）时，每个输出元素对应一次连续扫描：起点由
// 非归约维的 stride 累加得到。这样转置/permute 视图不必先拷成连续张量——
// 归约组本身仍交给上面那些内核，以 (outer=1, inner=1, axis) 的形式调用，
// 所以累加 dtype、NaN 传播、首个极值下标等语义与连续路径共用同一份实现。
//
// 被归约轴跨步（stride != 1）时该路径不适用：按列读是缓存最坏的形态，一次
// 连续拷贝反而更便宜——由 ops::iter 的策略决定物化，而不是这里硬判。
// ================================================================

struct ReduceMap {
    std::array<int64_t, kMaxTensorDims> dim{};
    std::array<int64_t, kMaxTensorDims> stride{};
    int     ndim  = 0;
    int64_t numel = 1;

    static ReduceMap make(const Tensor& a, int axis) {
        ReduceMap m;
        const auto shape = a.shape();
        const auto strides = a.strides();
        for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
            if (d == axis) continue;
            m.dim[m.ndim]    = shape[d];
            m.stride[m.ndim] = strides[d];
            ++m.ndim;
        }
        for (int i = 0; i < m.ndim; ++i) m.numel *= m.dim[i];
        return m;
    }

    int64_t base(int64_t index) const {
        int64_t off = 0;
        for (int d = ndim; d-- > 0;) {
            off += (index % dim[d]) * stride[d];
            index /= dim[d];
        }
        return off;
    }
};

// per_output(j, base)：输出第 j 个元素，输入起点偏移 base（元素，相对 Tensor::data()）。
// 返回 false 表示这个输入不吃跨步路径。
template <class F>
bool strided_reduce(const Tensor& a, int axis, F&& per_output) {
    if (a.is_contiguous() || a.strides()[axis] != 1) return false;
    const ReduceMap map = ReduceMap::make(a, axis);
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < map.numel; ++j)
        per_output(j, map.base(j));
    return true;
}

Tensor SumImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    auto a_shape = a.shape();
    int dim = a_shape.size();
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape(axis);
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];


    Tensor result(new_shape, reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    sum_kernel<T, R>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
                }))
                return;
            sum_kernel<T, R>(a_ptr, res_ptr, outer_size, inner_size, axis_size);
        });
    });
    return result;
}
Tensor MinImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    auto a_shape = a.shape();
    int dim = a_shape.size();

    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                min_kernel<T>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
            }))
            return;
        min_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor MaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    auto a_shape = a.shape();
    int dim = a_shape.size();

    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];

    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                max_kernel<T>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
            }))
            return;
        max_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor MeanImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    auto a_shape = a.shape();
    int dim = a_shape.size();

    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];

    Tensor result(new_shape, float_reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    mean_kernel<T, R>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
                }))
                return;
            mean_kernel<T, R>(a_ptr, res_ptr, outer_size, inner_size, axis_size);
        });
    });
    return result;
}
// ================================================================
bool AllImpl<Device::CPU>::execute(const Tensor& a, float value) {
    bool all;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        all = all_kernel<T>(a_ptr, value, a.numel());
    });
    return all;
}
bool AnyImpl<Device::CPU>::execute(const Tensor& a, float value) {
    bool any;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        any = any_kernel<T>(a_ptr, value, a.numel());
    });
    return any;
}
// ================================================================
Tensor ArgMaxImpl<Device::CPU>::execute(const Tensor &a, int axis){
    // 移除 a.shape(axis) 所在的轴
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    // 依旧需要计算 坐标 映射的函数
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a.shape()[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a.shape()[i];

    Tensor result(new_shape,DataType::INT32,Device::CPU);
    int32_t* res_ptr = static_cast<int32_t*>(result.data());
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                argmax_kernel<T>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
            }))
            return;
        argmax_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor ArgMinImpl<Device::CPU>::execute(const Tensor &a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < static_cast<int>(a.shape().size()); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 1D 输入移除唯一维度后为空；Tensor 不支持 0 维，退化为 {1}
    if (new_shape.empty()) new_shape = {1};
    auto a_shape = a.shape();
    int dim = a_shape.size();

    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];

    Tensor result(new_shape, DataType::INT32, Device::CPU);
    int32_t* res_ptr = static_cast<int32_t*>(result.data());
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                argmin_kernel<T>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
            }))
            return;
        argmin_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
// ================================================================
float ProdImpl<Device::CPU>::execute(const Tensor& a) {
    float prod = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        prod = prod_kernel<T>(static_cast<const T*>(a.data()), a.numel());
    });
    return prod;
}
Tensor ProdImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dim; ++i)
        if (i != axis) new_shape.push_back(a_shape[i]);
    if (new_shape.empty()) new_shape = {1};
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a_shape[axis];
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(new_shape, reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    prod_kernel<T, R>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size));
                }))
                return;
            prod_kernel<T, R>(a_ptr, res_ptr, outer_size, inner_size, axis_size);
        });
    });
    return result;
}
// ================================================================
float VarImpl<Device::CPU>::execute(const Tensor& a) {
    float var = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        var = var_std_full<T, false>(static_cast<const T*>(a.data()), a.numel());
    });
    return var;
}
Tensor VarImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dim; ++i)
        if (i != axis) new_shape.push_back(a_shape[i]);
    if (new_shape.empty()) new_shape = {1};
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a_shape[axis];
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(new_shape, float_reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            const T* a_ptr = static_cast<const T*>(a.data());
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    var_std_kernel<T, R, false>(a_ptr + base, res_ptr + j, 1, 1,
                                                int(axis_size));
                }))
                return;
            var_std_kernel<T, R, false>(a_ptr, res_ptr, outer_size, inner_size, axis_size);
        });
    });
    return result;
}
// ================================================================
float StdImpl<Device::CPU>::execute(const Tensor& a) {
    float std_ = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        std_ = var_std_full<T, true>(static_cast<const T*>(a.data()), a.numel());
    });
    return std_;
}
Tensor StdImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dim; ++i)
        if (i != axis) new_shape.push_back(a_shape[i]);
    if (new_shape.empty()) new_shape = {1};
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a_shape[axis];
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(new_shape, float_reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            const T* a_ptr = static_cast<const T*>(a.data());
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    var_std_kernel<T, R, true>(a_ptr + base, res_ptr + j, 1, 1,
                                               int(axis_size));
                }))
                return;
            var_std_kernel<T, R, true>(a_ptr, res_ptr, outer_size, inner_size, axis_size);
        });
    });
    return result;
}
// ================================================================
float NormImpl<Device::CPU>::execute(const Tensor& a, float p) {
    float norm = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        norm = norm_full<T>(static_cast<const T*>(a.data()), a.numel(), p);
    });
    return norm;
}
Tensor NormImpl<Device::CPU>::execute(const Tensor& a, float p, int axis) {
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dim; ++i)
        if (i != axis) new_shape.push_back(a_shape[i]);
    if (new_shape.empty()) new_shape = {1};
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a_shape[axis];
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(new_shape, float_reduction_type(a.dtype()), Device::CPU);
    const DataType out_type = result.dtype();
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        dispatch_dtype(out_type, [&](auto rid) {
            using R = typename decltype(rid)::type;
            const T* a_ptr = static_cast<const T*>(a.data());
            R* res_ptr = static_cast<R*>(result.data());
            if (strided_reduce(a, axis, [&](int64_t j, int64_t base) {
                    norm_kernel<T, R>(a_ptr + base, res_ptr + j, 1, 1, int(axis_size), p);
                }))
                return;
            norm_kernel<T, R>(a_ptr, res_ptr, outer_size, inner_size, axis_size, p);
        });
    });
    return result;
}
// ================================================================
Tensor CumsumImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    auto a_shape = a.shape();
    int dim = a_shape.size();
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a_shape[axis];
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i) inner_size *= a_shape[i];
    Tensor result(a.shape(), a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        cumsum_kernel<T>(static_cast<const T*>(a.data()), static_cast<T*>(result.data()),
                         outer_size, axis_size, inner_size);
    });
    return result;
}
// ================================================================
}  // namespace ops
