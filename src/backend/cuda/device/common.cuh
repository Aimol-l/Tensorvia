// CUDA 翻译单元共享的词汇表：启动几何、对库内 DataType 的 dtype 派发，以及契约
// 的标量辅助函数。
//
// 本目录下每个 .cu 文件都属于 header-first 产物，与宿主代码共享同一批生成头文件，
// 因此不再涉及任何 ABI id。
#ifndef TVV_CUDA_COMMON_CUH
#define TVV_CUDA_COMMON_CUH

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_launch.hpp"
#include "tvv_cuda_contract.cuh"

namespace tvv::cuda {

inline constexpr int kThreads = 256;

// 扁平启动的 block 数。至少为一个：零 block 启动是非法配置，其粘性错误会在下次
// 调用时浮现。
inline int blocks_for(int64_t numel, int threads = kThreads) {
    if (numel <= 0) return 1;
    const int64_t blocks = (numel + threads - 1) / threads;
    const int64_t limit  = std::numeric_limits<int>::max();
    return static_cast<int>(blocks > limit ? limit : blocks);
}

inline cudaError_t launch_status() {
    return cudaGetLastError();
}

// 宿主侧把 DataType 派发到接受设备标量类型的可调用对象。用本地标签而非
// std::type_identity，以保证设备单元能在不同 nvcc 前端间移植。
template <class T>
struct type_tag {
    using type = T;
};

template <class F>
void dispatch_dtype(DataType dtype, F&& f) {
    switch (dtype) {
        case DataType::BOOL:     f(type_tag<uint8_t>{}); break;
        case DataType::INT8:     f(type_tag<int8_t>{}); break;
        case DataType::INT16:    f(type_tag<int16_t>{}); break;
        case DataType::INT32:    f(type_tag<int32_t>{}); break;
        case DataType::INT64:    f(type_tag<int64_t>{}); break;
        case DataType::BFLOAT16: f(type_tag<__nv_bfloat16>{}); break;
        case DataType::FLOAT16:  f(type_tag<__half>{}); break;
        case DataType::FLOAT32:  f(type_tag<float>{}); break;
        case DataType::FLOAT64:  f(type_tag<double>{}); break;
        default: break;
    }
}

// 无 ABI 的标量参数以 double 传递；在宿主上转换一次。
template <class T>
TVV_CUDA_HD T scalar_from_double(double v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(static_cast<float>(v));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(static_cast<float>(v));
    } else {
        return static_cast<T>(v);
    }
}

// 把标量读入计算类型再写回。16 位浮点在计算中拓宽为 float，写出时经 double
// 舍入。
template <class C, class T>
TVV_CUDA_HD C to_compute(T v) {
    if constexpr (std::is_same_v<C, T>) {
        return v;
    } else if constexpr (contract::is_low_precision<T>::value) {
        return static_cast<C>(contract::load_as_float(v));
    } else {
        return static_cast<C>(v);
    }
}

template <class R, class C>
TVV_CUDA_HD R store_as(C v) {
    if constexpr (std::is_same_v<R, C>) {
        return v;
    } else if constexpr (contract::is_low_precision<R>::value) {
        return contract::store_from_double<R>(static_cast<double>(v));
    } else {
        return static_cast<R>(v);
    }
}

// std::numeric_limits 的成员是宿主 constexpr 函数，nvcc 在未加
// --expt-relaxed-constexpr 时拒绝在设备代码中调用。这些字面量变体覆盖归约内核
// 用到的累加类型。
template <class T>
TVV_CUDA_HD T max_finite() {
    if constexpr (std::is_same_v<T, float>)       return FLT_MAX;
    else if constexpr (std::is_same_v<T, double>) return DBL_MAX;
    else if constexpr (sizeof(T) == 8)            return static_cast<T>(INT64_MAX);
    else                                          return static_cast<T>(INT32_MAX);
}

template <class T>
TVV_CUDA_HD T lowest_finite() {
    if constexpr (std::is_same_v<T, float>)       return -FLT_MAX;
    else if constexpr (std::is_same_v<T, double>) return -DBL_MAX;
    else if constexpr (sizeof(T) == 8)            return static_cast<T>(INT64_MIN);
    else                                          return static_cast<T>(INT32_MIN);
}

template <class T>
TVV_CUDA_HD T inf_value() {
    return static_cast<T>(INFINITY);
}

// 参数结构体按值传递，使用时只依赖其成员；这些断言只是拦住意外的非平凡布局。
static_assert(sizeof(Bcast) % alignof(int64_t) == 0, "bcast layout");
static_assert(sizeof(Repack) % alignof(int64_t) == 0, "repack layout");

}  // namespace tvv::cuda

#endif  // TVV_CUDA_COMMON_CUH
