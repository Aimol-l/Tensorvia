// Reduction kernels — global and per-axis sum/min/max/mean, argmax/argmin,
// all/any and the NonZero count — plus the host driver that defines the
// corresponding Device::CUDA implementations.
#include "common.cuh"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace tvv::cuda {
namespace {

// One block reduces the whole buffer. The block loops over the input and then
// folds the per-thread partials in shared memory.
template <class T, class Out>
__global__ void reduce_all_kernel(const T* in, Out* out, int64_t numel, int op) {
    using Acc = contract::reduction_acc_t<T>;
    __shared__ Acc sdata[kThreads];
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    Acc acc;
    if (op == kReduceSum) {
        acc = Acc(0);
        for (int64_t i = idx; i < numel; i += stride) acc += static_cast<Acc>(to_compute<Acc>(in[i]));
    } else if (op == kReduceMin) {
        acc = max_finite<Acc>();
        for (int64_t i = idx; i < numel; i += stride)
            acc = contract::min_contract(acc, to_compute<Acc>(in[i]));
    } else {
        acc = lowest_finite<Acc>();
        for (int64_t i = idx; i < numel; i += stride)
            acc = contract::max_contract(acc, to_compute<Acc>(in[i]));
    }
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            Acc a = sdata[threadIdx.x];
            const Acc b = sdata[threadIdx.x + s];
            if (op == kReduceSum) sdata[threadIdx.x] = a + b;
            else if (op == kReduceMin) sdata[threadIdx.x] = contract::min_contract(a, b);
            else sdata[threadIdx.x] = contract::max_contract(a, b);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = store_as<Out>(sdata[0]);
}

// One thread per (outer, inner) pair walks the reduced axis.
template <class T, class Out>
__global__ void reduce_axis_kernel(const T* in, Out* out, ReduceAxis p, int op) {
    using Acc = contract::reduction_acc_t<T>;
    const int64_t total = p.outer * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t outer_idx = idx / p.inner;
        const int64_t inner_idx = idx % p.inner;
        const int64_t base = outer_idx * p.axis * p.inner + inner_idx;

        Acc acc;
        if (op == kReduceSum || op == kReduceMean) {
            acc = Acc(0);
            for (int64_t i = 0; i < p.axis; ++i) acc += static_cast<Acc>(to_compute<Acc>(in[base + i * p.inner]));
        } else if (op == kReduceMin) {
            acc = to_compute<Acc>(in[base]);
            for (int64_t i = 1; i < p.axis; ++i)
                acc = contract::min_contract(acc, to_compute<Acc>(in[base + i * p.inner]));
        } else {
            acc = to_compute<Acc>(in[base]);
            for (int64_t i = 1; i < p.axis; ++i)
                acc = contract::max_contract(acc, to_compute<Acc>(in[base + i * p.inner]));
        }
        if (op == kReduceMean) {
            out[idx] = store_as<Out>(static_cast<double>(acc) / static_cast<double>(p.axis));
        } else {
            out[idx] = store_as<Out>(acc);
        }
    }
}

template <class T>
__global__ void arg_reduce_kernel(const T* in, int32_t* out, ReduceAxis p, int is_max) {
    const int64_t total = p.outer * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t outer_idx = idx / p.inner;
        const int64_t inner_idx = idx % p.inner;
        const int64_t base = outer_idx * p.axis * p.inner + inner_idx;

        T best = in[base];
        int64_t best_idx = 0;
        for (int64_t i = 1; i < p.axis; ++i) {
            const T v = in[base + i * p.inner];
            const bool better = is_max ? contract::argmax_better(v, best)
                                       : contract::argmin_better(v, best);
            if (better) { best = v; best_idx = i; }
        }
        out[idx] = static_cast<int32_t>(best_idx);
    }
}

// All/Any: each block folds its slice into one flag, then combines into the
// single device word with an atomic and/or. The host launches init_kernel first
// on the same stream, so the destination never needs host-side zeroing.
template <class T>
__global__ void all_any_kernel(const T* in, unsigned int* out, int64_t numel, int is_all,
                               double value) {
    const float val = static_cast<float>(value);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    bool ok = is_all ? true : false;
    for (int64_t i = idx; i < numel; i += stride) {
        const bool hit = contract::match_value<T>(in[i], val);
        ok = is_all ? (ok && hit) : (ok || hit);
    }
    __shared__ int sdata[kThreads];
    sdata[threadIdx.x] = ok ? 1 : 0;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (is_all) sdata[threadIdx.x] &= sdata[threadIdx.x + s];
            else        sdata[threadIdx.x] |= sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (is_all) atomicAnd(out, static_cast<unsigned int>(sdata[0]));
        else        atomicOr(out, static_cast<unsigned int>(sdata[0]));
    }
}

__global__ void init_uint_kernel(unsigned int* out, unsigned int v) { *out = v; }
__global__ void init_ull_kernel(unsigned long long* out, unsigned long long v) { *out = v; }

template <class T>
__global__ void nonzero_count_kernel(const T* in, unsigned long long* out, int64_t numel) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (int64_t i = idx; i < numel; i += stride) {
        if (in[i] != T(0)) ++local;
    }
    if (local != 0) atomicAdd(out, local);
}

}  // namespace

cudaError_t launch_reduce_all(cudaStream_t stream, const void* in, void* out,
                              int64_t numel, DataType in_dtype, DataType out_dtype,
                              int op) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    // A single block, so the partial buffer is exactly one element.
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using Out = typename decltype(dst_tag)::type;
            reduce_all_kernel<T, Out><<<1, kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<Out*>(out), numel, op);
            status = launch_status();
        });
    });
    return status;
}

cudaError_t launch_reduce_axis(cudaStream_t stream, const void* in, void* out,
                               DataType in_dtype, DataType out_dtype, int op,
                               const ReduceAxis& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    const int64_t total = p.outer * p.inner;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using Out = typename decltype(dst_tag)::type;
            reduce_axis_kernel<T, Out><<<blocks_for(total), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<Out*>(out), p, op);
            status = launch_status();
        });
    });
    return status;
}

cudaError_t launch_arg_reduce(cudaStream_t stream, const void* in, void* out,
                              DataType dtype, bool is_max, const ReduceAxis& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    const int64_t total = p.outer * p.inner;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        arg_reduce_kernel<T><<<blocks_for(total), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<int32_t*>(out), p, is_max ? 1 : 0);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_all_any(cudaStream_t stream, const void* in, void* out,
                           int64_t numel, DataType dtype, bool is_all, double value) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    init_uint_kernel<<<1, 1, 0, stream>>>(static_cast<unsigned int*>(out),
                                          is_all ? 1u : 0u);
    cudaError_t status = launch_status();
    if (status != cudaSuccess) return status;
    status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        all_any_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<unsigned int*>(out), numel,
            is_all ? 1 : 0, value);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_nonzero_count(cudaStream_t stream, const void* in, void* out,
                                 int64_t numel, DataType dtype) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    init_ull_kernel<<<1, 1, 0, stream>>>(static_cast<unsigned long long*>(out), 0ull);
    cudaError_t status = launch_status();
    if (status != cudaSuccess) return status;
    status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        nonzero_count_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<unsigned long long*>(out), numel);
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

std::shared_ptr<CUDAContext> context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    return ctx;
}

// The kernels index a flat row-major buffer, so a view is materialized once
// here. Tensor::data() (not the impl's base pointer) is passed on, so a
// contiguous view with a non-zero offset reads its own elements.
Tensor materialize(const Tensor& a) {
    return a.is_contiguous() ? a : a.clone();
}

struct AxisLayout {
    tvv::cuda::ReduceAxis params{};
    std::vector<int64_t> shape;
};

AxisLayout axis_layout(const Tensor& a, int axis, const char* what) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error(std::string("cuda ") + what + ": axis out of range");

    AxisLayout layout;
    layout.params.axis  = a.shape(axis);
    layout.params.inner = 1;
    for (int i = axis + 1; i < dim; ++i) layout.params.inner *= a.shape(i);
    layout.params.outer = 1;
    for (int i = 0; i < axis; ++i) layout.params.outer *= a.shape(i);

    for (int i = 0; i < dim; ++i) {
        if (i != axis) layout.shape.push_back(a.shape(i));
    }
    // Dropping the only dimension of a 1D input leaves rank 0, which tensors do
    // not support; {1} matches the CPU/Vulkan fallback.
    if (layout.shape.empty()) layout.shape = {1};
    return layout;
}

// Global reductions fold into one FLOAT64 element on the device and are
// narrowed to float on the host, which is the API's return type.
double reduce_all_value(const Tensor& a, int op, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    Tensor out({1}, DataType::FLOAT64, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_reduce_all(static_cast<cudaStream_t>(ctx->stream()),
                                     src.data(), out_impl->data(),
                                     static_cast<int64_t>(src.numel()), src.dtype(),
                                     DataType::FLOAT64, op),
        what);
    ctx->wait();

    out.to_host();
    return static_cast<const double*>(out.data())[0];
}

Tensor axis_reduce(const Tensor& a, int axis, DataType out_type, int op, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    const AxisLayout layout = axis_layout(a, axis, what);

    Tensor out(layout.shape, out_type, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_reduce_axis(static_cast<cudaStream_t>(ctx->stream()),
                                      src.data(), out_impl->data(), src.dtype(),
                                      out_type, op, layout.params),
        what);
    ctx->wait();
    return out;
}

Tensor arg_reduce(const Tensor& a, int axis, bool is_max, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    const AxisLayout layout = axis_layout(a, axis, what);

    Tensor out(layout.shape, DataType::INT32, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_arg_reduce(static_cast<cudaStream_t>(ctx->stream()),
                                     src.data(), out_impl->data(), src.dtype(),
                                     is_max, layout.params),
        what);
    ctx->wait();
    return out;
}

bool all_any_value(const Tensor& a, float value, bool is_all, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    // The kernel initializes this word itself (init launch, then atomics), so
    // it does not need a host round-trip before the reduction.
    Tensor out({1}, DataType::INT32, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_all_any(static_cast<cudaStream_t>(ctx->stream()), src.data(),
                                  out_impl->data(), static_cast<int64_t>(src.numel()),
                                  src.dtype(), is_all, static_cast<double>(value)),
        what);
    ctx->wait();

    out.to_host();
    return static_cast<const int32_t*>(out.data())[0] != 0;
}

}  // namespace

float SumImpl<Device::CUDA>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::cuda::kReduceSum, "Sum"));
}

Tensor SumImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    // Sum widens narrow integers and 16-bit floats to a real accumulator type;
    // the Vulkan backend uses the same rule (the CPU one keeps the input dtype).
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::cuda::kReduceSum, "Sum");
}

float MinImpl<Device::CUDA>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::cuda::kReduceMin, "Min"));
}

Tensor MinImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::cuda::kReduceMin, "Min");
}

float MaxImpl<Device::CUDA>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::cuda::kReduceMax, "Max"));
}

Tensor MaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::cuda::kReduceMax, "Max");
}

float MeanImpl<Device::CUDA>::execute(const Tensor& a) {
    // Divide in double: above 2^24 a float divisor is no longer exact.
    const double sum = reduce_all_value(a, tvv::cuda::kReduceSum, "Mean");
    return static_cast<float>(sum / static_cast<double>(a.numel()));
}

Tensor MeanImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, float_reduction_type(a.dtype()),
                       tvv::cuda::kReduceMean, "Mean");
}

Tensor ArgMaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return arg_reduce(a, axis, true, "Argmax");
}

Tensor ArgMinImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return arg_reduce(a, axis, false, "Argmin");
}

bool AllImpl<Device::CUDA>::execute(const Tensor& a, float value) {
    return all_any_value(a, value, true, "All");
}

bool AnyImpl<Device::CUDA>::execute(const Tensor& a, float value) {
    return all_any_value(a, value, false, "Any");
}

size_t NonZeroImpl<Device::CUDA>::execute(const Tensor& a) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, "NonZero");
    Tensor out({1}, DataType::INT64, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl) throw std::runtime_error("cuda NonZero: failed to allocate the result");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_nonzero_count(static_cast<cudaStream_t>(ctx->stream()),
                                        src.data(), out_impl->data(),
                                        static_cast<int64_t>(src.numel()), src.dtype()),
        "NonZero launch");
    ctx->wait();

    out.to_host();
    return static_cast<size_t>(static_cast<const int64_t*>(out.data())[0]);
}

}  // namespace ops
