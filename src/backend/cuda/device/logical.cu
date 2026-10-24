// Broadcast and scalar comparison kernels, and the Equal/NotEqual/Greater/
// Less/GreaterEqual/LessEqual device implementations that drive them.
//
// The host promotes both operands to a common dtype, so each launch dispatches
// one scalar type.
#include "common.cuh"

#include "interface/backend/cuda/cuda_ops.hpp"
#include "interface/ops.hpp"
#include "interface/tensor.hpp"
#include "interface/types.hpp"

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;
using tvv::cuda::scalar_from_double;
using tvv::cuda::to_compute;
using tvv::cuda::type_tag;

namespace contract = tvv::cuda::contract;

// Same decode as the arithmetic kernels: accumulate each side's element offset
// with its own broadcast stride array.
TVV_CUDA_HD int64_t bcast_offset(int64_t idx, const tvv::cuda::Bcast& p,
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

template <class T>
TVV_CUDA_HD int8_t cmp_value(int op, T x, T y, double abs_tol, double rel_tol) {
    switch (op) {
        case tvv::cuda::kCmpEqual:        return contract::cmp_close(x, y, abs_tol, rel_tol) ? 1 : 0;
        case tvv::cuda::kCmpNotEqual:     return contract::cmp_close(x, y, abs_tol, rel_tol) ? 0 : 1;
        case tvv::cuda::kCmpGreater:      return x > y ? 1 : 0;
        case tvv::cuda::kCmpLess:         return x < y ? 1 : 0;
        case tvv::cuda::kCmpGreaterEqual: return x >= y ? 1 : 0;
        case tvv::cuda::kCmpLessEqual:    return x <= y ? 1 : 0;
        default:                          return 0;
    }
}

template <class T>
__global__ void compare_kernel(const T* a, const T* b, int8_t* out, tvv::cuda::Bcast p,
                               int op, double abs_tol, double rel_tol) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        const int64_t ia = p.off_a + bcast_offset(i, p, p.sa);
        const int64_t ib = p.off_b + bcast_offset(i, p, p.sb);
        out[p.off_o + i] = cmp_value(op, a[ia], b[ib], abs_tol, rel_tol);
    }
}

template <class T>
__global__ void compare_scalar_kernel(const T* a, int8_t* out, int64_t numel, int op,
                                      double scalar, double abs_tol, double rel_tol) {
    // The scalar arrives as double; convert once so half arithmetic stays in
    // the same native type the tensor uses.
    const T s = scalar_from_double<T>(scalar);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel; i += stride) {
        out[i] = cmp_value(op, a[i], s, abs_tol, rel_tol);
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_compare(cudaStream_t stream, const void* a, const void* b,
                           void* out, DataType dtype_a, DataType dtype_b, int op,
                           double tol_abs, double tol_rel, const Bcast& p) {
    if (a == nullptr || b == nullptr || out == nullptr) return cudaErrorInvalidValue;
    // The host promotes both sides, so a mixed pair here is a host-side bug.
    if (dtype_a != dtype_b) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype_a, [&](auto tag) {
        using T = typename decltype(tag)::type;
        compare_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            static_cast<const T*>(a), static_cast<const T*>(b),
            static_cast<int8_t*>(out), p, op, tol_abs, tol_rel);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_compare_scalar(cudaStream_t stream, const void* a, void* out,
                                  int64_t numel, DataType dtype, int op,
                                  double scalar, double tol_abs, double tol_rel) {
    if (a == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        compare_scalar_kernel<T><<<blocks_for(numel), kThreads, 0, stream>>>(
            static_cast<const T*>(a), static_cast<int8_t*>(out), numel, op, scalar,
            tol_abs, tol_rel);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace ops {

Tensor EqualImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpEqual, a, b);
}
Tensor NotEqualImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpNotEqual, a, b);
}
Tensor GreaterImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpGreater, a, b);
}
Tensor LessImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpLess, a, b);
}
Tensor GreaterEqualImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpGreaterEqual, a, b);
}
Tensor LessEqualImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    return cuda_ufunc::compare_impl(tvv::cuda::kCmpLessEqual, a, b);
}

Tensor EqualImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpEqual, a, val);
}
Tensor NotEqualImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpNotEqual, a, val);
}
Tensor GreaterImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpGreater, a, val);
}
Tensor LessImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpLess, a, val);
}
Tensor GreaterEqualImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpGreaterEqual, a, val);
}
Tensor LessEqualImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    return cuda_ufunc::compare_scalar(tvv::cuda::kCmpLessEqual, a, val);
}

}  // namespace ops
