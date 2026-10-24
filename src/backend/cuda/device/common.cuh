// Shared vocabulary for the CUDA translation units: launch geometry, dtype
// dispatch over the library's DataType, and the contract's scalar helpers.
//
// Every .cu file in this directory is part of the header-first artifact and
// shares the same generated headers as the host code, so no ABI ids are
// involved any more.
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

// Blocks for a flat launch. Always at least one: a zero-block launch is an
// invalid configuration and its sticky error would surface at the next call.
inline int blocks_for(int64_t numel, int threads = kThreads) {
    if (numel <= 0) return 1;
    const int64_t blocks = (numel + threads - 1) / threads;
    const int64_t limit  = std::numeric_limits<int>::max();
    return static_cast<int>(blocks > limit ? limit : blocks);
}

inline cudaError_t launch_status() {
    return cudaGetLastError();
}

// Host-side dispatch of a DataType to a callable taking a device scalar type.
// A local tag rather than std::type_identity keeps the device units portable
// across nvcc front ends.
template <class T>
struct type_tag {
    using type = T;
};

template <class F>
void dispatch_dtype(DataType dtype, F&& f) {
    switch (dtype) {
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

// ABI-free scalar parameters travel as double; convert them once on the host.
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

// Read a scalar into the compute type and write it back. 16-bit floats widen
// to float for computation and round through double on the way out.
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

// std::numeric_limits members are host constexpr functions, which nvcc refuses
// to call from device code without --expt-relaxed-constexpr. These literal
// variants cover the accumulator types the reduction kernels use.
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

// Parameter structs are passed by value and relied upon only through their
// members; these assertions just catch an accidental non-trivial layout.
static_assert(sizeof(Bcast) % alignof(int64_t) == 0, "bcast layout");
static_assert(sizeof(Repack) % alignof(int64_t) == 0, "repack layout");

}  // namespace tvv::cuda

#endif  // TVV_CUDA_COMMON_CUH
