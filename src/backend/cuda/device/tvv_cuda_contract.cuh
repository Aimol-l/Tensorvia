#ifndef TVV_CUDA_CONTRACT_CUH
#define TVV_CUDA_CONTRACT_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#define TVV_CUDA_HD __host__ __device__ inline

namespace tvv::cuda::contract {

template <class T>
struct is_half : std::false_type {};
template <>
struct is_half<__half> : std::true_type {};

template <class T>
struct is_bfloat : std::false_type {};
template <>
struct is_bfloat<__nv_bfloat16> : std::true_type {};

template <class T>
struct is_low_precision : std::bool_constant<is_half<T>::value || is_bfloat<T>::value> {};

// Compute type of one scalar: 16-bit floats widen to float, everything else is
// already its own compute type.
template <class T>
using promoted_t = std::conditional_t<is_low_precision<T>::value, float, T>;

// Compute type of a unary transcendental. The CPU backend calls std::sin(int)
// and friends, which select the double overload, so integer inputs are
// computed in double before the result is narrowed to the output dtype.
template <class T>
using unary_compute_t =
    std::conditional_t<std::is_integral<T>::value, double, promoted_t<T>>;

// Promotion of a binary pair. 16-bit floats join the float family; the wider
// of the two operands wins otherwise.
template <class A, class B>
using promote2_t = std::conditional_t<
    is_low_precision<A>::value || is_low_precision<B>::value, float,
    std::conditional_t<(sizeof(A) >= sizeof(B)), A, B>>;

template <class T>
TVV_CUDA_HD float load_as_float(T v) {
    if constexpr (is_half<T>::value) {
        return __half2float(v);
    } else if constexpr (is_bfloat<T>::value) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

template <class T>
TVV_CUDA_HD double load_as_double(T v) {
    if constexpr (is_half<T>::value) {
        return static_cast<double>(__half2float(v));
    } else if constexpr (is_bfloat<T>::value) {
        return static_cast<double>(__bfloat162float(v));
    } else {
        return static_cast<double>(v);
    }
}

// Host-side conversion back into a device scalar type.
template <class T>
TVV_CUDA_HD T store_from_double(double v) {
    if constexpr (is_half<T>::value) {
        return __float2half(static_cast<float>(v));
    } else if constexpr (is_bfloat<T>::value) {
        return __float2bfloat16(static_cast<float>(v));
    } else {
        return static_cast<T>(v);
    }
}

// Comparison tolerance tiers. Mirrors the CPU/Vulkan contract: doubles compare
// tightly, 16-bit floats loosely, because their spacing is ~1e-3 / ~1e-2.
template <class T>
TVV_CUDA_HD double tolerance_abs() {
    if constexpr (std::is_same_v<T, double>) {
        return 1e-9;
    } else if constexpr (is_half<T>::value) {
        return 1e-3;
    } else if constexpr (is_bfloat<T>::value) {
        return 1e-2;
    } else if constexpr (std::is_floating_point_v<T>) {
        return 1e-5;
    } else {
        return 0.0;
    }
}

template <class T>
TVV_CUDA_HD double tolerance_rel() {
    if constexpr (std::is_same_v<T, double>) {
        return 1e-12;
    } else if constexpr (is_half<T>::value) {
        return 1e-3;
    } else if constexpr (is_bfloat<T>::value) {
        return 1e-2;
    } else if constexpr (std::is_floating_point_v<T>) {
        return 1e-6;
    } else {
        return 0.0;
    }
}

template <class T>
TVV_CUDA_HD bool approximately_equal(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
        return a == b;
    } else {
        const double da = load_as_double(a);
        const double db = load_as_double(b);
        const double diff = da > db ? da - db : db - da;
        const double scale = da > db ? da : db;
        const double ref = scale < 0.0 ? -scale : scale;
        return diff <= tolerance_abs<T>() + tolerance_rel<T>() * ref;
    }
}

// Ordering predicates. NaN never compares greater or less, and equality must
// stay exact for integers (the approximate form is only for float families).
template <class T>
TVV_CUDA_HD bool is_nan(T v) {
    if constexpr (std::is_floating_point_v<T>) {
        return v != v;
    } else if constexpr (is_half<T>::value) {
        return __half2float(v) != __half2float(v);
    } else if constexpr (is_bfloat<T>::value) {
        return __bfloat162float(v) != __bfloat162float(v);
    } else {
        return false;
    }
}

template <class T>
TVV_CUDA_HD T relu_contract(T v) {
    // NaN is a signal, not a value to clamp: NumPy's maximum/clip and PyTorch's
    // relu/clamp all propagate it. Integers have no NaN, so the test folds away.
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    return v > T(0) ? v : T(0);
}

template <class T>
TVV_CUDA_HD T clamp_contract(T v, T lo, T hi) {
    // Same propagation rule as relu_contract.
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    const T clamped_low = v < lo ? lo : v;
    return clamped_low > hi ? hi : clamped_low;
}

// Accumulator for sum-style reductions. Narrow integers and 16-bit floats must
// not wrap or lose bits before the result is written.
template <class T>
using reduction_acc_t = std::conditional_t<
    std::is_integral<T>::value,
    std::conditional_t<(sizeof(T) <= 2), int32_t,
                       std::conditional_t<(sizeof(T) == 4), int64_t, int64_t>>,
    promoted_t<T>>;

// min/max propagate NaN: it marks a computation that went wrong, and silently
// replacing it with a finite bound is a plausible-looking wrong answer.
template <class T>
TVV_CUDA_HD T min_contract(T a, T b) {
    if (is_nan(a)) return a;
    if (is_nan(b)) return b;
    return b < a ? b : a;
}

template <class T>
TVV_CUDA_HD T max_contract(T a, T b) {
    if (is_nan(a)) return a;
    if (is_nan(b)) return b;
    return b > a ? b : a;
}

// argmax/argmin: the first NaN wins; ties keep the smaller index, so a later
// value only replaces the incumbent when it compares strictly greater/less.
template <class T>
TVV_CUDA_HD bool argmax_better(T val, T best) {
    if (is_nan(best)) return false;
    if (is_nan(val)) return true;
    return val > best;
}

template <class T>
TVV_CUDA_HD bool argmin_better(T val, T best) {
    if (is_nan(best)) return false;
    if (is_nan(val)) return true;
    return val < best;
}

// All/Any compare against a float scalar. Integers compare exactly; float
// families use a dtype-dependent absolute tolerance.
template <class T>
TVV_CUDA_HD double match_tolerance() {
    if constexpr (std::is_same_v<T, double>) return 1e-10;
    else if constexpr (std::is_same_v<T, float>) return 1e-5;
    else if constexpr (is_half<T>::value) return 1e-3;
    else if constexpr (is_bfloat<T>::value) return 1e-2;
    else return 0.0;
}

template <class T>
TVV_CUDA_HD bool match_value(T x, float val) {
    if constexpr (std::is_integral_v<T>) {
        return x == static_cast<T>(val);
    } else {
        const double d = load_as_double(x) - static_cast<double>(val);
        return (d < 0.0 ? -d : d) <= match_tolerance<T>();
    }
}

// Equality test used by Equal/NotEqual. The tolerances arrive from the host
// because promotion to a common dtype can widen 16-bit operands, losing the
// precision tier the comparison contract is supposed to use.
template <class T>
TVV_CUDA_HD bool cmp_close(T a, T b, double abs_tol, double rel_tol) {
    if constexpr (std::is_integral_v<T>) {
        return a == b;
    } else {
        const double da = load_as_double(a);
        const double db = load_as_double(b);
        const double diff = da > db ? da - db : db - da;
        const double scale = (da > db ? da : db);
        const double magnitude = scale < 0.0 ? -scale : scale;
        const double rel = rel_tol * magnitude;
        return diff <= (rel > abs_tol ? rel : abs_tol);
    }
}

// Division contract: floating point follows IEEE (x/0 is ±inf), integer
// division by zero is defined as 0 instead of the C++ undefined behavior.
template <class T>
TVV_CUDA_HD T div_contract(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
        return y == T(0) ? T(0) : static_cast<T>(x / y);
    } else {
        return x / y;
    }
}

}  // namespace tvv::cuda::contract

#endif  // TVV_CUDA_CONTRACT_CUH
