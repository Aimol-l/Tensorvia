// Shared vocabulary for the SYCL device units.
//
// These units are compiled by the DPC++ compiler, not by the project's host
// compiler, and they only ever see sycl_island.h. Nothing here allocates or
// throws across the seam: the launch entry points catch device-side failures
// and report the status code defined by the island header.
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

// The queue lives here so every unit can recover it from the opaque handle.
struct Context {
    ::sycl::queue queue;
    Context()
        : queue(::sycl::default_selector_v, ::sycl::property::queue::in_order()) {}
};

inline ::sycl::queue& queue_of(void* ctx) {
    return static_cast<Context*>(ctx)->queue;
}

// ---------------------------------------------------------------------------
// Scalar vocabulary. sycl::half and oneAPI bfloat16 are storage types: every
// arithmetic and reduction path widens them to float first.
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

// Narrow integers and 16-bit floats must not wrap or lose bits before the
// result is written.
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
// Numeric contract, mirrored from the CPU/CUDA backends.
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
// dtype dispatch.
// ---------------------------------------------------------------------------

template <class T>
struct type_tag {
    using type = T;
};

// `f` returns an int status; the first matching arm wins. Every dtype is
// instantiated, so a lambda with `if constexpr` inside remains cheap.
template <class F>
inline int dispatch_dtype(int dtype, F&& f) {
    switch (dtype) {
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

// Decode a flat output index against the broadcast shape, accumulating each
// operand's element offset with its own (already broadcast) stride array.
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

// 64-bit hash used by Random. Self-contained: no oneDPL or <random> in the
// device units, and no per-element generator state to allocate.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform [0, 1), 53 bits of mantissa.
inline double uniform01(uint64_t seed, int64_t index) {
    const uint64_t bits = splitmix64(seed ^ (static_cast<uint64_t>(index) *
                                             0x9E3779B97F4A7C15ull));
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Kernel submission helpers. The queue is in-order, so successive submissions
// from one host thread never need explicit dependencies.
// ---------------------------------------------------------------------------

inline ::sycl::range<1> range_of(int64_t n) {
    return ::sycl::range<1>(static_cast<size_t>(n < 0 ? 0 : n));
}

}  // namespace tvv::sycl_dev

#endif  // TVV_SYCL_DEVICE_COMMON_H
