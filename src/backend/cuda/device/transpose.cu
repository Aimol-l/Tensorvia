// 通用 N 维置换，输出连续。二维转置就是 {1,0} 的情形；同一个内核也覆盖 batch
// 与更高秩。
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

template <class T>
__global__ void transpose_kernel(const T* in, T* out, Transpose p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        // 用输入步长解码连续输入下标，再在输出中编码置换后的坐标。
        int64_t rem = i;
        int64_t coord[kMaxDims];
        for (int d = 0; d < p.ndim; ++d) {
            const int64_t s = p.in_strides[d];
            coord[d] = s > 0 ? rem / s : 0;
            if (s > 0) rem %= s;
        }
        int64_t out_off = 0;
        for (int d = 0; d < p.ndim; ++d) out_off += coord[p.axes[d]] * p.out_strides[d];
        out[out_off] = in[i];
    }
}

}  // namespace

cudaError_t launch_transpose(cudaStream_t stream, const void* in, void* out,
                             DataType dtype, const Transpose& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (p.ndim <= 0 || p.ndim > kMaxDims) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        transpose_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), p);
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

// 把 `a`（连续，可能带视图 offset）置换进 `out`，后者必须已具备置换后的形状与
// 连续 buffer。
void launch_transpose(const Tensor& a, Tensor& out, const std::vector<int64_t>& axes,
                      const char* what) {
    const int ndim = static_cast<int>(a.shape().size());
    if (ndim <= 0 || ndim > tvv::cuda::kMaxDims)
        throw std::runtime_error(std::string("cuda ") + what + ": unsupported rank");
    if (!a.is_contiguous())
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": input must be contiguous (permute() is a view)");
    if (static_cast<int>(axes.size()) != ndim)
        throw std::runtime_error(std::string("cuda ") + what + ": axes size mismatch");

    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": destination must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(out_impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");

    tvv::cuda::Transpose p{};
    p.numel = static_cast<int64_t>(a.numel());
    p.ndim = ndim;
    for (int d = 0; d < ndim; ++d) {
        p.in_shape[d] = a.shape(d);
        p.in_strides[d] = a.strides(d);
        p.axes[d] = axes[d];
    }
    // 输出是全新分配（或经 reshape 得到）且连续：行主序。
    for (int d = ndim - 1; d >= 0; --d) {
        p.out_strides[d] = (d == ndim - 1) ? 1 : p.out_strides[d + 1] * out.shape(d + 1);
    }

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_transpose(static_cast<cudaStream_t>(ctx->stream()), a.data(),
                                    out.data(), a.dtype(), p),
        what);
}

std::vector<int64_t> permuted_shape(const Tensor& a,
                                    std::initializer_list<int64_t> axes) {
    std::vector<int64_t> shape;
    shape.reserve(axes.size());
    for (int64_t ax : axes) shape.push_back(a.shape(ax));
    return shape;
}

}  // namespace

void TransposeImpl<Device::CUDA>::execute(Tensor& a) {
    // 公开 API 把这种形式限定为二维连续张量。
    Tensor out({a.shape(1), a.shape(0)}, a.dtype(), Device::CUDA);
    launch_transpose(a, out, {1, 0}, "Transpose");
    a = std::move(out);
}

Tensor TransposeImpl<Device::CUDA>::execute(const Tensor& a,
                                            std::initializer_list<int64_t> axes) {
    Tensor out(permuted_shape(a, axes), a.dtype(), Device::CUDA);
    launch_transpose(a, out, std::vector<int64_t>(axes), "Transpose");
    return out;
}

void TransposeImpl<Device::CUDA>::execute(const Tensor& a, Tensor& dst,
                                          std::initializer_list<int64_t> axes) {
    std::vector<int64_t> shape = permuted_shape(a, axes);
    dst = dst.reshape(shape);
    launch_transpose(a, dst, std::vector<int64_t>(axes), "Transpose");
}

}  // namespace ops
