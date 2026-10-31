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

// 单个标量的计算类型：16 位浮点拓宽到 float，其余类型本身就是自己的计算类型。
template <class T>
using promoted_t = std::conditional_t<is_low_precision<T>::value, float, T>;

// 一元超越算子的计算类型。CPU 后端调用 std::sin(int) 之类，会选中 double 重载，
// 所以整型输入先以 double 计算，再把结果收窄到输出 dtype。
template <class T>
using unary_compute_t =
    std::conditional_t<std::is_integral<T>::value, double, promoted_t<T>>;

// 二元配对的提升规则。16 位浮点并入 float 家族；其余情况两者中较宽者胜出。
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

// 宿主侧转换回设备标量类型。
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

// 比较容差档位。与 CPU/Vulkan 契约一致：double 比较得紧，16 位浮点比较得松，
// 因为它们的间距约为 1e-3 / 1e-2。
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

// 大小比较谓词。NaN 既不大于也不小于任何值；整型的相等必须保持精确（近似形式只
// 用于浮点家族）。
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
    // NaN 是信号，不是可裁剪的值：NumPy 的 maximum/clip 与 PyTorch 的 relu/clamp
    // 都会传播它。整型没有 NaN，所以该判断会被编译掉。
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    return v > T(0) ? v : T(0);
}

template <class T>
TVV_CUDA_HD T clamp_contract(T v, T lo, T hi) {
    // 与 relu_contract 相同的传播规则。
    if constexpr (!std::is_integral_v<T>) {
        if (is_nan(v)) return v;
    }
    const T clamped_low = v < lo ? lo : v;
    return clamped_low > hi ? hi : clamped_low;
}

// sum 类归约的累加器。窄整型与 16 位浮点在结果写出之前不得回绕或丢位。
template <class T>
using reduction_acc_t = std::conditional_t<
    std::is_integral<T>::value,
    std::conditional_t<(sizeof(T) <= 2), int32_t,
                       std::conditional_t<(sizeof(T) == 4), int64_t, int64_t>>,
    promoted_t<T>>;

// min/max 传播 NaN：它标志着计算出错，悄悄用有限边界替换它只会得到一个看似合理
// 的错误答案。
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

// argmax/argmin：首个 NaN 胜出；并列时保留较小索引，所以后来的值只有在严格
// 大于/小于当前值时才取代它。
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

// All/Any 与浮点标量比较。整型精确比较；浮点家族使用依赖 dtype 的绝对容差。
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

// Equal/NotEqual 使用的相等判定。容差由宿主传入，因为提升到公共 dtype 可能拓宽
// 16 位操作数，从而丢失比较契约本应采用的那一档精度。
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

// 除法契约：浮点遵循 IEEE（x/0 为 ±inf），整型除零定义为 0，而不是 C++ 的未定义
// 行为。
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
