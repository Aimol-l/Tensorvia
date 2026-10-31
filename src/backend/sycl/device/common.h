// SYCL 设备单元共享的词汇表。
//
// 这些单元由 DPC++ 编译器编译，而非项目的宿主编译器，它们只能看到
// sycl_island.h。此处不会跨接缝分配内存或抛异常：启动入口捕获设备侧失败，并
// 报告 island 头文件中定义的状态码。
#ifndef TVV_SYCL_DEVICE_COMMON_H
#define TVV_SYCL_DEVICE_COMMON_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "sycl_island.h"

namespace tvv::sycl_dev {

using tvv::sycl::kMaxDims;

inline constexpr size_t kWorkGroup = 256;

// queue 放在这里，每个单元都能从那个不透明句柄取回它。索引 -1 表示“由
// default_selector_v 决定”；非负索引是 sycl::device::get_devices() 中的位置，
// 也就是外部设备探测所报告的那个索引空间。
struct Context {
    ::sycl::queue queue;
    explicit Context(int device_index = -1)
        : queue(device_of(device_index), ::sycl::property::queue::in_order()) {}

    static ::sycl::device device_of(int device_index) {
        if (device_index < 0) return ::sycl::device(::sycl::default_selector_v);
        const auto devices = ::sycl::device::get_devices();
        if (device_index >= static_cast<int>(devices.size())) {
            throw ::sycl::exception(::sycl::errc::invalid,
                                    "no SYCL device at the requested index");
        }
        return devices[static_cast<size_t>(device_index)];
    }
};

inline ::sycl::queue& queue_of(void* ctx) {
    return static_cast<Context*>(ctx)->queue;
}

// ---------------------------------------------------------------------------
// 标量词汇表。sycl::half 与 oneAPI bfloat16 是存储类型：所有算术与归约路径都
// 先把它们拓宽为 float。
// ---------------------------------------------------------------------------

template <class T>
struct is_half : std::false_type {};
template <>
struct is_half<::sycl::half> : std::true_type {};

template <class T>
struct is_bfloat : std::false_type {};
template <>
struct is_bfloat<::sycl::ext::oneapi::bfloat16> : std::true_type {};

template <class T>
struct is_low_precision : std::bool_constant<is_half<T>::value || is_bfloat<T>::value> {};

template <class T>
using promoted_t = std::conditional_t<is_low_precision<T>::value, float, T>;

template <class T>
using unary_compute_t = std::conditional_t<std::is_integral_v<T>, double, promoted_t<T>>;

// 窄整型与 16 位浮点在结果写出之前不得回绕或丢位。
template <class T>
using reduction_acc_t = std::conditional_t<
    std::is_integral_v<T>,
    std::conditional_t<sizeof(T) <= 2, int32_t, int64_t>,
    promoted_t<T>>;

template <class T>
inline float load_as_float(T v) {
    return static_cast<float>(v);
}

template <class T>
inline double load_as_double(T v) {
    return static_cast<double>(v);
}

template <class R>
inline R store_from_double(double v) {
    if constexpr (is_half<R>::value) {
        return R(static_cast<float>(v));
    } else if constexpr (is_bfloat<R>::value) {
        return R(static_cast<float>(v));
    } else {
        return static_cast<R>(v);
    }
}

template <class C, class T>
inline C to_compute(T v) {
    if constexpr (std::is_same_v<C, T>) {
        return v;
    } else if constexpr (is_low_precision<T>::value) {
        return static_cast<C>(load_as_float(v));
    } else {
        return static_cast<C>(v);
    }
}

template <class R, class C>
inline R store_as(C v) {
    if constexpr (std::is_same_v<R, C>) {
        return v;
    } else if constexpr (is_low_precision<R>::value) {
        return store_from_double<R>(static_cast<double>(v));
    } else {
        return static_cast<R>(v);
    }
}

template <class T>
inline T max_finite() {
    if constexpr (std::is_same_v<T, float>)       return 3.402823466e+38F;
    else if constexpr (std::is_same_v<T, double>) return 1.7976931348623157e+308;
    else if constexpr (sizeof(T) == 8)            return static_cast<T>(INT64_MAX);
    else                                          return static_cast<T>(INT32_MAX);
}

template <class T>
inline T lowest_finite() {
    if constexpr (std::is_same_v<T, float>)       return -3.402823466e+38F;
    else if constexpr (std::is_same_v<T, double>) return -1.7976931348623157e+308;
    else if constexpr (sizeof(T) == 8)            return static_cast<T>(INT64_MIN);
    else                                          return static_cast<T>(INT32_MIN);
}

template <class T>
inline T inf_value() {
    if constexpr (std::is_same_v<T, double>) return __builtin_huge_val();
    else                                     return __builtin_huge_valf();
}

// ---------------------------------------------------------------------------
// 数值契约，镜像自 CPU/CUDA 后端。
// ---------------------------------------------------------------------------

namespace contract {

template <class T>
inline double tolerance_abs() {
    if constexpr (std::is_same_v<T, double>) return 1e-9;
    else if constexpr (is_half<T>::value)    return 1e-3;
    else if constexpr (is_bfloat<T>::value)  return 1e-2;
    else if constexpr (std::is_floating_point_v<T>) return 1e-5;
    else                                     return 0.0;
}

template <class T>
inline double tolerance_rel() {
    if constexpr (std::is_same_v<T, double>) return 1e-12;
    else if constexpr (is_half<T>::value)    return 1e-3;
    else if constexpr (is_bfloat<T>::value)  return 1e-2;
    else if constexpr (std::is_floating_point_v<T>) return 1e-6;
    else                                     return 0.0;
}

template <class T>
inline bool is_nan(T v) {
    if constexpr (std::is_integral_v<T>) {
        return false;
    } else if constexpr (is_low_precision<T>::value) {
        const float f = load_as_float(v);
        return f != f;
    } else {
        return v != v;
    }
}

template <class T>
inline T relu_contract(T v) {
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    return v > T(0) ? v : T(0);
}

template <class T>
inline T clamp_contract(T v, T lo, T hi) {
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    const T clamped_low = v < lo ? lo : v;
    return clamped_low > hi ? hi : clamped_low;
}

template <class T>
inline T min_contract(T a, T b) {
    if (is_nan(a)) return a;
    if (is_nan(b)) return b;
    return b < a ? b : a;
}

template <class T>
inline T max_contract(T a, T b) {
    if (is_nan(a)) return a;
    if (is_nan(b)) return b;
    return b > a ? b : a;
}

template <class T>
inline bool argmax_better(T val, T best) {
    if (is_nan(best)) return false;
    if (is_nan(val)) return true;
    return val > best;
}

template <class T>
inline bool argmin_better(T val, T best) {
    if (is_nan(best)) return false;
    if (is_nan(val)) return true;
    return val < best;
}

template <class T>
inline double match_tolerance() {
    if constexpr (std::is_same_v<T, double>) return 1e-10;
    else if constexpr (std::is_same_v<T, float>) return 1e-5;
    else if constexpr (is_half<T>::value)   return 1e-3;
    else if constexpr (is_bfloat<T>::value) return 1e-2;
    else return 0.0;
}

template <class T>
inline bool match_value(T x, float val) {
    if constexpr (std::is_integral_v<T>) {
        return x == static_cast<T>(val);
    } else {
        const double d = load_as_double(x) - static_cast<double>(val);
        return (d < 0.0 ? -d : d) <= match_tolerance<T>();
    }
}

template <class T>
inline bool cmp_close(T a, T b, double abs_tol, double rel_tol) {
    if constexpr (std::is_integral_v<T>) {
        return a == b;
    } else {
        const double da = load_as_double(a);
        const double db = load_as_double(b);
        const double diff = da > db ? da - db : db - da;
        const double scale = da > db ? da : db;
        const double magnitude = scale < 0.0 ? -scale : scale;
        const double rel = rel_tol * magnitude;
        return diff <= (rel > abs_tol ? rel : abs_tol);
    }
}

template <class T>
inline T div_contract(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
        return y == T(0) ? T(0) : static_cast<T>(x / y);
    } else {
        return x / y;
    }
}

}  // namespace contract

// ---------------------------------------------------------------------------
// dtype 派发。
// ---------------------------------------------------------------------------

template <class T>
struct type_tag {
    using type = T;
};

// `f` 返回 int 状态码；第一个匹配的分支生效。所有 dtype 都会被实例化，因此内部
// 带 `if constexpr` 的 lambda 依然廉价。
template <class F>
inline int dispatch_dtype(int dtype, F&& f) {
    switch (dtype) {
        case tvv::sycl::kBool:     return f(type_tag<uint8_t>{});
        case tvv::sycl::kInt8:     return f(type_tag<int8_t>{});
        case tvv::sycl::kInt16:    return f(type_tag<int16_t>{});
        case tvv::sycl::kInt32:    return f(type_tag<int32_t>{});
        case tvv::sycl::kInt64:    return f(type_tag<int64_t>{});
        case tvv::sycl::kBfloat16: return f(type_tag<::sycl::ext::oneapi::bfloat16>{});
        case tvv::sycl::kFloat16:  return f(type_tag<::sycl::half>{});
        case tvv::sycl::kFloat32:  return f(type_tag<float>{});
        case tvv::sycl::kFloat64:  return f(type_tag<double>{});
        default:                   return tvv::sycl::kSyclUnsupported;
    }
}

// 按广播形状解码扁平输出下标，用各自（已完成广播的）步长数组累加每个操作数的
// 元素 offset。
inline int64_t bcast_offset(int64_t idx, const tvv::sycl::Bcast& p,
                            const int64_t* strides) {
    int64_t off = 0;
    for (int d = p.ndim - 1; d >= 0; --d) {
        const int64_t dim = p.shape[d];
        if (dim <= 0) continue;
        off += (idx % dim) * strides[d];
        idx /= dim;
    }
    return off;
}

// Random 使用的 64 位哈希。自包含：设备单元中不用 oneDPL 或 <random>，也无需
// 为每元素生成器状态分配内存。
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// 均匀分布在 [0, 1)，尾数 53 位。
inline double uniform01(uint64_t seed, int64_t index) {
    const uint64_t bits = splitmix64(seed ^ (static_cast<uint64_t>(index) *
                                             0x9E3779B97F4A7C15ull));
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// 内核提交辅助。队列是 in-order 的，因此同一宿主线程的连续提交永远不需要显式
// 依赖。
// ---------------------------------------------------------------------------

inline ::sycl::range<1> range_of(int64_t n) {
    return ::sycl::range<1>(static_cast<size_t>(n < 0 ? 0 : n));
}

}  // namespace tvv::sycl_dev

#endif  // TVV_SYCL_DEVICE_COMMON_H
