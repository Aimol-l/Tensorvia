// Concatenation along one axis, plus the host driver for ConcatImpl.
//
// The host entry point launches one kernel per input instead of staging a
// device-side pointer array, so no managed memory or temporary allocation is
// involved.
#include "common.cuh"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace {

using tvv::cuda::kMaxDims;

struct ConcatOne {
    int64_t numel;
    int64_t axis_offset;
    int32_t ndim;
    int32_t axis;
    int64_t in_shape[kMaxDims];
    int64_t in_strides[kMaxDims];
    int64_t out_strides[kMaxDims];
};

// Decode the input's flat index, then re-encode it in the output, shifting the
// coordinate on the concatenation axis by this input's prefix offset.
template <class T>
__global__ void concat_kernel(const T* in, T* out, ConcatOne p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        int64_t rem = i;
        int64_t in_off = 0;
        int64_t out_off = 0;
        for (int d = p.ndim - 1; d >= 0; --d) {
            const int64_t dim = p.in_shape[d];
            const int64_t c = dim > 0 ? rem % dim : 0;
            if (dim > 0) rem /= dim;
            in_off += c * p.in_strides[d];
            const int64_t oc = (d == p.axis) ? c + p.axis_offset : c;
            out_off += oc * p.out_strides[d];
        }
        out[out_off] = in[in_off];
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_concat(cudaStream_t stream, void* out,
                          const void* const* inputs, const int64_t* input_offsets,
                          int32_t num_inputs, DataType dtype,
                          const int64_t* out_strides, const int64_t* in_shapes,
                          const int64_t* in_strides, int32_t ndim, int32_t axis) {
    if (out == nullptr || inputs == nullptr || input_offsets == nullptr)
        return cudaErrorInvalidValue;
    if (num_inputs <= 0 || ndim <= 0 || ndim > kMaxDims) return cudaErrorInvalidValue;
    if (axis < 0 || axis >= ndim) return cudaErrorInvalidValue;

    cudaError_t status = cudaSuccess;
    for (int32_t n = 0; n < num_inputs; ++n) {
        ConcatOne p{};
        p.ndim = ndim;
        p.axis = axis;
        p.axis_offset = input_offsets[n];
        p.numel = 1;
        for (int d = 0; d < ndim; ++d) {
            p.in_shape[d] = in_shapes[static_cast<int64_t>(n) * ndim + d];
            p.in_strides[d] = in_strides[static_cast<int64_t>(n) * ndim + d];
            p.out_strides[d] = out_strides[d];
            p.numel *= p.in_shape[d];
        }
        if (inputs[n] == nullptr) return cudaErrorInvalidValue;
        status = cudaErrorNotSupported;
        tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            concat_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
                static_cast<const T*>(inputs[n]), static_cast<T*>(out), p);
            status = launch_status();
        });
        if (status != cudaSuccess) return status;
    }
    return status;
}

}  // namespace tvv::cuda

namespace ops {

Tensor ConcatImpl<Device::CUDA>::execute(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::runtime_error("Concat: empty inputs");
    if (tensors.size() > static_cast<size_t>(tvv::cuda::kMaxDims))
        throw std::runtime_error("Concat: too many inputs");
    const int ndim = static_cast<int>(tensors[0].shape().size());
    if (dim < 0 || dim >= ndim) throw std::runtime_error("Concat: invalid axis");
    if (ndim > tvv::cuda::kMaxDims) throw std::runtime_error("Concat: too many dimensions");
    const DataType dtype = tensors[0].dtype();

    std::vector<int64_t> out_shape(tensors[0].shape().begin(), tensors[0].shape().end());
    out_shape[dim] = 0;
    for (const auto& t : tensors) out_shape[dim] += t.shape(dim);

    Tensor out(out_shape, dtype, Device::CUDA);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl) throw std::runtime_error("cuda Concat: failed to allocate the result");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(out_impl->context());
    if (!ctx) throw std::runtime_error("cuda Concat: missing CUDAContext");

    const size_t n = tensors.size();
    std::vector<const void*> inputs(n);
    std::vector<int64_t> offsets(n);
    std::vector<int64_t> in_shapes(n * static_cast<size_t>(ndim));
    std::vector<int64_t> in_strides(n * static_cast<size_t>(ndim));
    int64_t prefix = 0;
    for (size_t i = 0; i < n; ++i) {
        // tensor.data() includes the view offset, so a contiguous slice of a
        // larger buffer concatenates its own elements.
        inputs[i] = tensors[i].data();
        offsets[i] = prefix;
        prefix += tensors[i].shape(dim);
        for (int d = 0; d < ndim; ++d) {
            in_shapes[i * ndim + d] = tensors[i].shape(d);
            in_strides[i * ndim + d] = tensors[i].strides(d);
        }
    }

    std::vector<int64_t> out_strides(out.strides().begin(), out.strides().end());
    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_concat(static_cast<cudaStream_t>(ctx->stream()), out.data(),
                                 inputs.data(), offsets.data(), static_cast<int32_t>(n),
                                 dtype, out_strides.data(), in_shapes.data(),
                                 in_strides.data(), ndim, dim),
        "Concat launch");
    ctx->wait();
    return out;
}

}  // namespace ops
