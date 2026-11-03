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

// tiled 转置：32x32 的共享内存块，读合并、写合并。
//
// 通用内核每个线程读一个连续输入元素、按输出步长散写到全局——写一侧的访存完全
// 无法合并（同一 warp 的 32 个元素相隔一行长度）。这里先把一个 tile 收进
// shared，再从 shared 转置着写出，两边的全局访存都落在连续地址上；padding 的
// +1 列错开 shared 的 bank。
//
// 形状参数按"批次 x 行 x 列"给：输入一行 `in_row` 个元素、输出一行
// `out_row` 个元素，行列两轴的基址步长各自传入。这样二维转置是 batches = 1 的
// 特例，三维里"输入连续轴 × 输出连续轴"成十字的置换也直接复用同一个内核。
constexpr int kTileDim  = 32;
constexpr int kTileRows = 8;

template <class T>
__global__ void transpose_tiled_kernel(const T* __restrict__ in, T* __restrict__ out,
                                       int64_t rows, int64_t cols, int64_t in_batch,
                                       int64_t in_row, int64_t out_batch,
                                       int64_t out_row) {
    __shared__ T tile[kTileDim][kTileDim + 1];

    const int64_t in_base = blockIdx.z * in_batch;
    const int64_t out_base = blockIdx.z * out_batch;

    const int64_t in_x = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.x;
    const int64_t in_y = static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kTileRows) {
        if (in_x < cols && in_y + j < rows)
            tile[threadIdx.y + j][threadIdx.x] = in[in_base + (in_y + j) * in_row + in_x];
    }
    __syncthreads();

    const int64_t out_x = static_cast<int64_t>(blockIdx.y) * kTileDim + threadIdx.x;
    const int64_t out_y = static_cast<int64_t>(blockIdx.x) * kTileDim + threadIdx.y;
    for (int j = 0; j < kTileDim; j += kTileRows) {
        if (out_x < rows && out_y + j < cols)
            out[out_base + (out_y + j) * out_row + out_x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// 输出连续轴与输入连续轴同向的 rank-3 置换：每个 (batch, row) 是一段沿列方向
// 的连续拷贝，不需要交换行列。用一维网格沿列铺线程即可保持合并访存。
template <class T>
__global__ void transpose_copy_kernel(const T* __restrict__ in, T* __restrict__ out,
                                      int64_t rows, int64_t cols, int64_t in_batch,
                                      int64_t in_row, int64_t out_batch,
                                      int64_t out_row) {
    const int64_t x = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= cols) return;
    const int64_t y = blockIdx.y;
    const int64_t b = blockIdx.z;
    out[b * out_batch + y * out_row + x] = in[b * in_batch + y * in_row + x];
}

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
    // 2-D 交换两轴是绝大多数调用（含 Transpose(a) 的原地形式）。该几何下输入
    // 连续，输出连续，可以按 tile 双向合并访存。
    const bool swap2d = p.ndim == 2 && p.axes[0] == 1 && p.axes[1] == 0;

    // rank-3 的 tiled 退化。输入连续轴是第 2 维；tile 取第 2 维与 axes[2]
    // （输出侧步长为 1 的那条输入轴），剩下的轴当批次：
    //   axes[2] == 2 —— 输出连续轴与输入连续轴同向，逐行连续拷贝；
    //   否则        —— 两个连续方向成十字，走共享 tile 的转置。
    // 六种三维置换都落进这两类。
    enum class Plan { kGeneric, kCopy, kTranspose };
    Plan plan = swap2d ? Plan::kTranspose : Plan::kGeneric;
    int64_t rows = 0, cols = 0, batches = 1;
    int64_t in_batch = 0, in_row = 0, out_batch = 0, out_row = 0;
    if (swap2d) {
        rows = p.in_shape[0];
        cols = p.in_shape[1];
        in_row = cols;
        out_row = rows;
    } else if (p.ndim == 3 && p.in_strides[2] == 1) {
        int64_t out_stride_of[3] = {};
        for (int d = 0; d < 3; ++d) out_stride_of[p.axes[d]] = p.out_strides[d];
        const int64_t ya = p.axes[2];  // 输出侧步长为 1 的输入轴
        const int64_t ba = ya == 2 ? p.axes[0] : 3 - ya - 2;
        if (ya == 2) {
            const int64_t ra = p.axes[1];
            plan = Plan::kCopy;
            rows = p.in_shape[ra];
            cols = p.in_shape[2];
            in_batch = p.in_strides[ba];
            in_row = p.in_strides[ra];
            out_batch = out_stride_of[ba];
            out_row = out_stride_of[ra];
        } else {
            plan = Plan::kTranspose;
            rows = p.in_shape[ya];
            cols = p.in_shape[2];
            in_batch = p.in_strides[ba];
            in_row = p.in_strides[ya];
            out_batch = out_stride_of[ba];
            out_row = out_stride_of[2];
        }
        batches = p.in_shape[ba];
    }
    // grid.y / grid.z 有 65535 上限；空维度在核心层已被拦掉，这里再兜一层，
    // 避免发出 0 大小的 grid。
    if (plan != Plan::kGeneric) {
        const int64_t gy = plan == Plan::kCopy ? rows : (rows + kTileDim - 1) / kTileDim;
        if (rows <= 0 || cols <= 0 || batches <= 0 || gy > 65535 || batches > 65535)
            plan = Plan::kGeneric;
    }

    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (plan == Plan::kCopy) {
            const int threads = cols < 256 ? static_cast<int>((cols + 31) / 32) * 32 : 256;
            const dim3 grid(static_cast<unsigned>((cols + threads - 1) / threads),
                            static_cast<unsigned>(rows), static_cast<unsigned>(batches));
            transpose_copy_kernel<T><<<grid, threads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), rows, cols, in_batch,
                in_row, out_batch, out_row);
        } else if (plan == Plan::kTranspose) {
            const dim3 threads(kTileDim, kTileRows);
            const dim3 grid(static_cast<unsigned>((cols + kTileDim - 1) / kTileDim),
                            static_cast<unsigned>((rows + kTileDim - 1) / kTileDim),
                            static_cast<unsigned>(batches));
            transpose_tiled_kernel<T><<<grid, threads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), rows, cols, in_batch,
                in_row, out_batch, out_row);
        } else {
            transpose_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out), p);
        }
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
