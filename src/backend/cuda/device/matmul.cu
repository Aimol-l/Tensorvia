// Tiled matrix multiplication and the host driver for MatMulImpl.
//
// A is viewed as [batch, rows, k] and B as [batch, k, cols]; a 2D call is the
// batch=1 case. The host promotes both operands and the destination to a
// common dtype and materializes contiguous buffers, so one scalar type per
// launch is enough here.
#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "interface/backend/cuda/cuda_context.hpp"
#include "interface/backend/cuda/cuda_ops.hpp"
#include "interface/backend/cuda/cuda_tensor.hpp"
#include "interface/ops.hpp"
#include "interface/tensor.hpp"
#include "interface/types.hpp"

#include "cuda_launch.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;

namespace contract = tvv::cuda::contract;

constexpr int kTile = 16;

// accum_t<T> 是 promoted_t<T> + promoted_t<T> 的结果类型。原写法用
// std::declval 在 decltype 里取类型，但 clang 的 CUDA 路线把 declval 标成
// __host__，禁止在 __global__ 里实例化（nvcc 没这个限制）。这里按标准
// 整型提升规则直接写：窄于 int 的整型提升为 int，其余就是 promoted_t<T>
// ——对 dispatch 覆盖的 8 种 dtype 逐一验证过与原写法一致。
template <class T>
using accum_t = std::conditional_t<
    std::is_integral<contract::promoted_t<T>>::value &&
        sizeof(contract::promoted_t<T>) < sizeof(int),
    int,
    contract::promoted_t<T>>;

template <class T>
__global__ void matmul_kernel(const T* a, const T* b, T* out,
                              int64_t batch, int64_t rows, int64_t cols, int64_t k) {
    __shared__ T tile_a[kTile][kTile + 1];
    __shared__ T tile_b[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    const int64_t bz = blockIdx.z;

    accum_t<T> sum = 0;
    const int num_tiles = static_cast<int>((k + kTile - 1) / kTile);
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int64_t a_k = static_cast<int64_t>(tile_idx) * kTile + tx;
        tile_a[ty][tx] = (row < rows && a_k < k)
                             ? a[bz * rows * k + static_cast<int64_t>(row) * k + a_k]
                             : T(0);
        const int64_t b_k = static_cast<int64_t>(tile_idx) * kTile + ty;
        tile_b[ty][tx] = (b_k < k && col < cols)
                             ? b[bz * k * cols + b_k * cols + col]
                             : T(0);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kTile; ++i) {
            sum += static_cast<accum_t<T>>(tile_a[ty][i]) *
                   static_cast<accum_t<T>>(tile_b[i][tx]);
        }
        __syncthreads();
    }
    if (row < rows && col < cols) {
        out[bz * rows * cols + static_cast<int64_t>(row) * cols + col] =
            contract::store_from_double<T>(static_cast<double>(sum));
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_matmul(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype, int64_t batch, int64_t rows,
                          int64_t cols, int64_t k) {
    if (a == nullptr || b == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (batch < 0 || rows < 0 || cols < 0 || k < 0) return cudaErrorInvalidValue;

    const dim3 threads(kTile, kTile);
    const dim3 blocks(static_cast<unsigned>(std::max<int64_t>(1, (cols + kTile - 1) / kTile)),
                      static_cast<unsigned>(std::max<int64_t>(1, (rows + kTile - 1) / kTile)),
                      static_cast<unsigned>(std::max<int64_t>(1, batch)));

    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        matmul_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out),
            batch, rows, cols, k);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------

namespace ops {

// [rows,k] @ [k,cols] -> [rows,cols]
// [batch,rows,k] @ [batch,k,cols] -> [batch,rows,cols]
Tensor MatMulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    const bool batched = a.shape().size() == 3;
    const int64_t batch  = batched ? a.shape(0) : 1;
    const int64_t rows   = batched ? a.shape(1) : a.shape(0);
    const int64_t common = batched ? a.shape(2) : a.shape(1);
    const int64_t cols   = batched ? b.shape(2) : b.shape(1);

    std::vector<int64_t> newshape =
        batched ? std::vector<int64_t>{batch, rows, cols}
                : std::vector<int64_t>{rows, cols};

    // Half-precision operands are widened to FLOAT32 first, exactly like the
    // CPU backend: the 16-bit formats are storage types, not accumulator types.
    Tensor a_storage, b_storage;
    const Tensor* a_p = &a;
    const Tensor* b_p = &b;
    if (a.dtype() == DataType::FLOAT16 || a.dtype() == DataType::BFLOAT16) {
        a_storage = ops::Typecast(a, DataType::FLOAT32);
        a_p = &a_storage;
    }
    if (b.dtype() == DataType::FLOAT16 || b.dtype() == DataType::BFLOAT16) {
        b_storage = ops::Typecast(b, DataType::FLOAT32);
        b_p = &b_storage;
    }

    const DataType res_type = compute_type(a_p->dtype(), b_p->dtype());
    Tensor dst(newshape, res_type, Device::CUDA);
    // The kernel reads contiguous row-major operands; dtype promotion goes
    // through Typecast (no-op when the dtype already matches).
    const Tensor A = a_p->dtype() == res_type ? a_p->contiguous() : ops::Typecast(*a_p, res_type);
    const Tensor B = b_p->dtype() == res_type ? b_p->contiguous() : ops::Typecast(*b_p, res_type);

    const auto A_impl   = std::dynamic_pointer_cast<CUDATensor>(A.get_impl());
    const auto B_impl   = std::dynamic_pointer_cast<CUDATensor>(B.get_impl());
    const auto dst_impl = std::dynamic_pointer_cast<CUDATensor>(dst.get_impl());
    if (!A_impl || !B_impl || !dst_impl)
        throw std::runtime_error("cuda MatMul: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(A_impl->context());
    if (!ctx) throw std::runtime_error("cuda MatMul: missing CUDAContext");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_matmul(static_cast<cudaStream_t>(ctx->stream()),
                                 A_impl->data(), B_impl->data(), dst_impl->data(),
                                 res_type, batch, rows, cols, common),
        "MatMul launch");
    ctx->wait();
    return dst;
}

}  // namespace ops
