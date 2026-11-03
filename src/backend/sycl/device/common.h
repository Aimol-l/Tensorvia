#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "sycl_island.h"

namespace tvv::sycl_dev {

using tvv::sycl::kMaxDims;

inline constexpr size_t kWorkGroup = 256;

// 启动函数的 catch 走这里：先把运行时的原始描述记到调用线程上（接缝只有纯 C
// 状态码，"device failure" 说明不了任何事），再返回失败。
inline int record_failure(const std::exception& e) {
    ::sycl_c_record_error(e.what());
    return tvv::sycl::kSyclFailure;
}

inline int record_failure() {
    ::sycl_c_record_error(nullptr);
    return tvv::sycl::kSyclFailure;
}

// 设备能不能真的跑我们的内核。
//
// 一台机器可以同时暴露几块“SYCL 设备”，但其中有的装不下我们的设备镜像：本机的
// NVIDIA 设备走 CUDA 适配器，运行时拿不到 fill_kernel<float>（"No kernel named
// ... was found"），Intel OpenCL CPU 设备则跑得动。索引空间若不筛掉这类设备，
// 调用方按序号逐个探测就会撞上一个永远失败的条目——那正是
// DeviceId.SynchronizeCoversAllIndices 一直在报的错。
//
// 探测用一个最小内核（分配 + 1 个元素的 parallel_for + wait），而不是查设备属性：
// 能失败的地方全在“这个运行时的镜像能不能落到这块设备上”，属性看不出来。
inline bool device_runs_kernels(const ::sycl::device& device) {
    int* buffer = nullptr;
    bool ok = false;
    try {
        ::sycl::queue queue{device,
                            ::sycl::property_list{::sycl::property::queue::in_order()}};
        buffer = ::sycl::malloc_device<int>(1, queue);
        if (buffer != nullptr) {
            queue.parallel_for(::sycl::range<1>{1}, [=](::sycl::id<1> i) { buffer[i] = 1; });
            queue.wait_and_throw();
            ok = true;
        }
        // 释放必须走分配它的那条队列（同一个 context），不能另造一个 context。
        if (buffer != nullptr) ::sycl::free(buffer, queue);
    } catch (...) {
        return false;
    }
    return ok;
}

// 可用设备在 sycl::device::get_devices() 里的下标，按原顺序；首次调用时探测一次
// 并缓存。全部探测失败时退回“全部可用”——此时是运行时的探测能力出了问题，
// 早退成 0 个设备会把整个后端藏起来，反而更难查。
inline const std::vector<int>& usable_device_indices() {
    static const std::vector<int> usable = [] {
        const auto devices = ::sycl::device::get_devices();
        std::vector<int> filtered;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (device_runs_kernels(devices[i])) filtered.push_back(static_cast<int>(i));
        }
        if (filtered.empty()) {
            for (size_t i = 0; i < devices.size(); ++i) filtered.push_back(static_cast<int>(i));
        }
        return filtered;
    }();
    return usable;
}

// 每条（线程, 上下文）一条 in-order 队列：首次使用时惰性创建，登记在 context
// 里供 wait 与析构使用。所有队列都由同一个 sycl::context 构造——USM 分配归属
// context，不共享 context 的队列之间不能互用同一块内存。
// 索引 -1 表示“默认设备”；非负索引是可用设备列表中的位置（见
// usable_device_indices），也就是外部设备探测报告的那个索引空间。
struct Context {
    ::sycl::device  device;
    ::sycl::context sycl_context;
    uint64_t        id = 0;
    std::mutex      queues_mutex;
    std::vector<::sycl::queue*> queues;  // 所有线程已创建的队列

    explicit Context(int device_index = -1);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // 调用线程在本 context 上的队列；首次调用时创建并登记到 queues。
    ::sycl::queue& current_queue();

    static ::sycl::device device_of(int device_index) {
        const auto devices = ::sycl::device::get_devices();
        const auto& usable = usable_device_indices();
        // 默认设备：可用设备里的第一块 GPU；没有 GPU 就取第一块可用的。
        if (device_index < 0) {
            for (int slot : usable) {
                if (devices[static_cast<size_t>(slot)].is_gpu())
                    return devices[static_cast<size_t>(slot)];
            }
            if (usable.empty()) {
                throw ::sycl::exception(::sycl::errc::invalid, "no usable SYCL device");
            }
            return devices[static_cast<size_t>(usable.front())];
        }
        if (device_index >= static_cast<int>(usable.size())) {
            throw ::sycl::exception(::sycl::errc::invalid,
                                    "no SYCL device at the requested index");
        }
        return devices[static_cast<size_t>(usable[static_cast<size_t>(device_index)])];
    }
};

inline ::sycl::queue& queue_of(void* ctx) {
    return static_cast<Context*>(ctx)->current_queue();
}

// 句柄就是 context 拥有的那个 queue 对象的地址，生命周期与 context 一致。
inline ::sycl::queue& queue_from_handle(void* handle) {
    return *static_cast<::sycl::queue*>(handle);
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