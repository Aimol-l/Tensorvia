// 填充 / 随机初始化内核，以及 Zeros/Ones/Fill/Random 的宿主驱动。
//
// Random 使用 cuRAND 的 Philox 计数器式生成器，并以元素下标作为子序列，因此无需
// 分配 curandState 数组：旧实现为每个元素分配 48 字节，还要在一次设备同步之后
// 才释放。种子始终来自宿主。
#include "common.cuh"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <curand_kernel.h>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

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
        // (seed, subsequence, offset)：下标即子序列，这让各条流彼此独立而无需
        // 任何共享状态。
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
    // BOOL 写入会规范化：任何非零填充值都表示真（1）。
    const double v = (dtype == DataType::BOOL) ? (value != 0.0 ? 1.0 : 0.0) : value;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto id) {
        using T = typename decltype(id)::type;
        fill_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<T*>(out), numel, scalar_from_double<T>(v));
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
// 宿主驱动。
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
        // const_cast：填充是唯一的宿主侧写入；这里用 Tensor::data()（感知 offset）
        // 而不是 TensorImpl::data()。
        tvv::cuda::launch_fill(static_cast<cudaStream_t>(target.ctx->stream()),
                               const_cast<void*>(t.data()),
                               static_cast<int64_t>(t.numel()), t.dtype(), value),
        what);
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
    return temp;
}

}  // namespace ops
