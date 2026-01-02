#include <print>
#include <omp.h>
#include <random>
#include <stdfloat>
#include <cstddef>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cuda_context.h" 
#include "backend/cuda/ops/initializer.h"
using namespace via;

namespace ops { 

template <typename T>
__global__ void fill_value_cuda(T* ptr, T val, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel)   ptr[i] = val;
}

template <typename T>
__global__ void init_curand_states(curandState* states, uint64_t seed, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    curand_init(seed, i, 0, &states[i]);
}

template <typename T>
__global__ void fill_random_cuda(T* ptr, curandState* states, size_t numel, float min, float max) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    curandState localState = states[i];
    float r = curand_uniform(&localState); // [0,1)
    r = min + r * (max - min);
    ptr[i] = static_cast<T>(r);
    states[i] = localState; // 保存状态
}

Tensor ZerosImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor tmp(shape, dtype, Device::CUDA);
    constexpr size_t threads = 256;
    size_t blocks = (tmp.numel() + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(tmp.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(dtype,tmp.data());
    std::visit([&](auto ptr) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
        if constexpr(std::is_same_v<T, float16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__half*>(tmp.data()), __half(0), tmp.numel());
        } else if constexpr(std::is_same_v<T, bfloat16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__nv_bfloat16*>(tmp.data()), __nv_bfloat16(0), tmp.numel());
        } else {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<T*>(tmp.data()), T(0), tmp.numel());
        }
    }, A);
    ctx_impl->wait();
    return tmp;
}
Tensor OnesImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor tmp(shape, dtype, Device::CUDA);
    constexpr size_t threads = 256;
    size_t blocks = (tmp.numel() + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(tmp.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(dtype, tmp.data());
    std::visit([&](auto ptr) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
        if constexpr(std::is_same_v<T, float16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__half*>(tmp.data()), __half(1), tmp.numel());
        } else if constexpr(std::is_same_v<T, bfloat16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__nv_bfloat16*>(tmp.data()), __nv_bfloat16(1), tmp.numel());
        } else {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<T*>(tmp.data()), T(1), tmp.numel());
        }
    }, A);
    ctx_impl->wait();
    return tmp;
}
Tensor FillImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype, float value){
    Tensor tmp(shape, dtype, Device::CUDA);
    constexpr size_t threads = 256;
    size_t blocks = (tmp.numel() + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(tmp.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(dtype, tmp.data());
    std::visit([&](auto ptr) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
        if constexpr(std::is_same_v<T, float16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__half*>(tmp.data()), __half(value), tmp.numel());
        } else if constexpr(std::is_same_v<T, bfloat16>) {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__nv_bfloat16*>(tmp.data()), __nv_bfloat16(value), tmp.numel());
        } else {
            fill_value_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<T*>(tmp.data()), T(value), tmp.numel());
        }
    }, A);
    ctx_impl->wait();
    return tmp;
}
Tensor RandomImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype,float min,float max){
    Tensor tmp(shape, dtype, Device::CUDA);
    auto numel = tmp.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(tmp.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(dtype, tmp.data());
    uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::visit([&](auto ptr) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(ptr)>>;
        curandState* d_states;
        cudaMalloc(&d_states, numel * sizeof(curandState));
        init_curand_states<T><<<blocks, threads>>>(d_states, seed, numel);
        if constexpr(std::is_same_v<T, float16>) {
            fill_random_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__half*>(tmp.data()), d_states, numel, min, max);
        } else if constexpr(std::is_same_v<T, bfloat16>) {
            fill_random_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<__nv_bfloat16*>(tmp.data()), d_states, numel, min, max);
        } else {
            fill_random_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<T*>(tmp.data()), d_states, numel, min, max);
        }
        ctx_impl->wait();
        cudaFree(d_states);
    }, A);
    return tmp;
}
template struct ZerosImpl<Device::CUDA>;
template struct OnesImpl<Device::CUDA>;
template struct FillImpl<Device::CUDA>;
template struct RandomImpl<Device::CUDA>;

}