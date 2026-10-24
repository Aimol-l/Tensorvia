// Fill / random initialization kernels and the host driver for
// Zeros/Ones/Fill/Random.
//
// Random uses cuRAND's Philox counter-based generator with a per-element
// subsequence, so no curandState array has to be allocated: the reference
// implementation allocated 48 bytes per element and freed it after a device
// synchronization. The seed always comes from the host.
#include "common.cuh"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <curand_kernel.h>

#include <tensorvia/cuda_context.h>
#include <tensorvia/cuda_ops.h>
#include <tensorvia/cuda_tensor.h>
#include <tensorvia/ops_api.h>
#include <tensorvia/tensor.h>
#include <tensorvia/types.h>

#include "cuda_launch.hpp"

namespace {

template <class T>
__global__ void fill_kernel(T* __restrict__ out, int64_t numel, T value) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = start; i < numel; i += stride) out[i] = value;
}

template <class T>
__global__ void random_kernel(T* __restrict__ out, int64_t numel, float lo, float hi,
                              uint64_t seed) {
    const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = start; i < numel; i += stride) {
        curandStatePhilox4_32_10_t state;
        // (seed, subsequence, offset): the index is the subsequence, which keeps
        // streams independent without any shared state.
        curand_init(seed, static_cast<unsigned long long>(i), 0, &state);
        const float r = curand_uniform(&state);  // (0, 1]
        out[i] = tvv::cuda::contract::store_from_double<T>(
            static_cast<double>(lo) + static_cast<double>(r) * (static_cast<double>(hi) - lo));
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_fill(cudaStream_t stream, void* out, int64_t numel,
                        DataType dtype, double value) {
    if (out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto id) {
        using T = typename decltype(id)::type;
        fill_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<T*>(out), numel, scalar_from_double<T>(value));
        status = launch_status();
    });
    return status;
}

cudaError_t launch_random(cudaStream_t stream, void* out, int64_t numel,
                          DataType dtype, float lo, float hi, uint64_t seed) {
    if (out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto id) {
        using T = typename decltype(id)::type;
        random_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<T*>(out), numel, lo, hi, seed);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

namespace ops {

namespace {

struct Target {
    std::shared_ptr<CUDATensor>  impl;
    std::shared_ptr<CUDAContext> ctx;
};

Target target_of(const Tensor& t, const char* what) {
    auto impl = std::dynamic_pointer_cast<CUDATensor>(t.get_impl());
    if (!impl) throw std::runtime_error(std::string("cuda ") + what + ": expected a CUDATensor");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    return {std::move(impl), std::move(ctx)};
}

void fill_tensor(const Tensor& t, double value, const char* what) {
    auto target = target_of(t, what);
    target.ctx->activate();
    tvv::cuda::check_cuda(
        // const_cast: filling is the one host-side write; Tensor::data() is the
        // offset-aware pointer, unlike TensorImpl::data().
        tvv::cuda::launch_fill(static_cast<cudaStream_t>(target.ctx->stream()),
                               const_cast<void*>(t.data()),
                               static_cast<int64_t>(t.numel()), t.dtype(), value),
        what);
    target.ctx->wait();
}

uint64_t make_seed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
}

}  // namespace

Tensor ZerosImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::CUDA);
    fill_tensor(temp, 0.0, "Zeros");
    return temp;
}

Tensor OnesImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::CUDA);
    fill_tensor(temp, 1.0, "Ones");
    return temp;
}

Tensor FillImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                       float value) {
    Tensor temp(shape, dtype, Device::CUDA);
    fill_tensor(temp, static_cast<double>(value), "Fill");
    return temp;
}

void FillImpl<Device::CUDA>::execute(Tensor& a, float value) {
    fill_tensor(a, static_cast<double>(value), "Fill");
}

Tensor RandomImpl<Device::CUDA>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                         float min, float max) {
    Tensor temp(shape, dtype, Device::CUDA);
    auto target = target_of(temp, "Random");
    target.ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_random(static_cast<cudaStream_t>(target.ctx->stream()),
                                 target.impl->data(), static_cast<int64_t>(temp.numel()),
                                 dtype, min, max, make_seed()),
        "Random");
    target.ctx->wait();
    return temp;
}

}  // namespace ops
