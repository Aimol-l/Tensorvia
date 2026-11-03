// 分块矩阵乘法，以及 MatMulImpl 的宿主驱动。
//
// A 视为 [batch, rows, k]，B 视为 [batch, k, cols]；二维调用就是 batch=1 的情形。
// 宿主把两个操作数和目标提升到公共 dtype 并物化为连续 buffer，所以这里每次启动只
// 需一种标量类型。
#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;

namespace contract = tvv::cuda::contract;

// 分块参数：一个 block 算 64x64 的输出 tile，每个线程算其中 8x8。
//
// 旧内核是 16x16、每线程 1 个输出：K 方向每读一个 A/B 元素只摊到一次 FMA，
// 且 shared 里没有寄存器级复用。现在每个 k 步读 8+8 个 shared 元素做 64 次
// FMA，算术强度提高到 4 FLOP/字节量级，借 shared 广播把带宽压力压下去。
//
// 128x128 tile（kBlockK 16 才能塞进 48 KB 静态 shared）实测更慢：每线程仍是
// 8x8、256 线程下 64 个累加器的寄存器压力把占用率压到 2 个 block/SM，1024
// 只有 8.7 TFLOPS。这里的几何是实测选出来的，不是回退的默认值。
//
// 输出小到 64x64 tile 不足 32 个时（256^3 只有 16 个），换下面的 16x16 内核，
// 用 block 数换并行度；512 及以上两个内核的差距已有 1.7 倍以上。
constexpr int kBlockM   = 64;
constexpr int kBlockN   = 64;
constexpr int kBlockK   = 32;
constexpr int kThreadM  = 8;
constexpr int kThreadN  = 8;
constexpr int kThreadsX = kBlockN / kThreadN;  // 8
constexpr int kThreadsY = kBlockM / kThreadM;  // 8
constexpr int kMatmulThreads = kThreadsX * kThreadsY;  // 64

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

// 一个 block 负责输出 tile [blockIdx.y*64, +64) x [blockIdx.x*64, +64)，
// blockIdx.z 是 batch。线程 (ty, tx) 负责 tile 内的 4x4 子块。
//
// 两个 shared tile 的装载下标都让"线程号最快的维度"落在连续的行内存上：
// A 沿 k 连续、B 沿 col 连续，因此每个 warp 读的是一段 128B 连续地址。
template <class T>
__global__ void matmul_kernel(const T* __restrict__ a, const T* __restrict__ b,
                              T* __restrict__ out, int64_t rows, int64_t cols,
                              int64_t k) {
    __shared__ T tile_a[kBlockK][kBlockM + 1];
    __shared__ T tile_b[kBlockK][kBlockN + 1];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int64_t z = blockIdx.z;
    const T* a_batch = a + z * rows * k;
    const T* b_batch = b + z * k * cols;
    T* c_batch = out + z * rows * cols;

    const int64_t block_row = static_cast<int64_t>(blockIdx.y) * kBlockM;
    const int64_t block_col = static_cast<int64_t>(blockIdx.x) * kBlockN;
    const int64_t row0 = block_row + threadIdx.y * kThreadM;
    const int64_t col0 = block_col + threadIdx.x * kThreadN;

    accum_t<T> acc[kThreadM][kThreadN];
#pragma unroll
    for (int i = 0; i < kThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) acc[i][j] = accum_t<T>(0);

    const int64_t k_tiles = (k + kBlockK - 1) / kBlockK;
    for (int64_t kt = 0; kt < k_tiles; ++kt) {
        const int64_t k0 = kt * kBlockK;

        // A: 1024 个元素（kBlockK * kBlockM），idx 的低位是 k → 合并读。
#pragma unroll
        for (int idx = tid; idx < kBlockK * kBlockM; idx += kMatmulThreads) {
            const int kk = idx % kBlockK;
            const int mm = idx / kBlockK;
            const int64_t gm = block_row + mm;
            const int64_t gk = k0 + kk;
            tile_a[kk][mm] = (gm < rows && gk < k) ? a_batch[gm * k + gk] : T(0);
        }
        // B: idx 的低位是输出列 → 合并读。
#pragma unroll
        for (int idx = tid; idx < kBlockK * kBlockN; idx += kMatmulThreads) {
            const int nn = idx % kBlockN;
            const int kk = idx / kBlockN;
            const int64_t gn = block_col + nn;
            const int64_t gk = k0 + kk;
            tile_b[kk][nn] = (gk < k && gn < cols) ? b_batch[gk * cols + gn] : T(0);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; ++kk) {
            accum_t<T> va[kThreadM];
            accum_t<T> vb[kThreadN];
#pragma unroll
            for (int i = 0; i < kThreadM; ++i)
                va[i] = static_cast<accum_t<T>>(tile_a[kk][threadIdx.y * kThreadM + i]);
#pragma unroll
            for (int j = 0; j < kThreadN; ++j)
                vb[j] = static_cast<accum_t<T>>(tile_b[kk][threadIdx.x * kThreadN + j]);
#pragma unroll
            for (int i = 0; i < kThreadM; ++i)
#pragma unroll
                for (int j = 0; j < kThreadN; ++j) acc[i][j] += va[i] * vb[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kThreadM; ++i) {
        const int64_t gr = row0 + i;
        if (gr >= rows) continue;
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
            const int64_t gc = col0 + j;
            if (gc >= cols) continue;
            c_batch[gr * cols + gc] =
                contract::store_from_double<T>(static_cast<double>(acc[i][j]));
        }
    }
}

}  // namespace

namespace tvv::cuda {

// 小输出尺寸用：16x16 tile、每线程一个输出。
//
// 64x64 tile 在 256x256 上只有 16 个 block，填不满 66 个 SM；这个几何有 256 个
// block，靠并行度取胜（实测 0.021 ms vs 0.039 ms）。复用同一个输出维度上界时
// 共享内存只有 1 KB，占用率不受限。
constexpr int kSmallTile = 16;

namespace {

template <class T>
__global__ void matmul_small_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                    T* __restrict__ out, int64_t rows, int64_t cols,
                                    int64_t k) {
    __shared__ T tile_a[kSmallTile][kSmallTile + 1];
    __shared__ T tile_b[kSmallTile][kSmallTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int64_t row = static_cast<int64_t>(blockIdx.y) * kSmallTile + ty;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kSmallTile + tx;
    const int64_t bz = blockIdx.z;

    accum_t<T> sum = accum_t<T>(0);
    const int num_tiles = static_cast<int>((k + kSmallTile - 1) / kSmallTile);
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int64_t a_k = static_cast<int64_t>(tile_idx) * kSmallTile + tx;
        tile_a[ty][tx] = (row < rows && a_k < k)
                             ? a[bz * rows * k + row * k + a_k]
                             : T(0);
        const int64_t b_k = static_cast<int64_t>(tile_idx) * kSmallTile + ty;
        tile_b[ty][tx] = (b_k < k && col < cols)
                             ? b[bz * k * cols + b_k * cols + col]
                             : T(0);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kSmallTile; ++i)
            sum += static_cast<accum_t<T>>(tile_a[ty][i]) *
                   static_cast<accum_t<T>>(tile_b[i][tx]);
        __syncthreads();
    }
    if (row < rows && col < cols)
        out[bz * rows * cols + row * cols + col] =
            contract::store_from_double<T>(static_cast<double>(sum));
}

}  // namespace

cudaError_t launch_matmul(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype, int64_t batch, int64_t rows,
                          int64_t cols, int64_t k) {
    if (a == nullptr || b == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (batch < 0 || rows < 0 || cols < 0 || k < 0) return cudaErrorInvalidValue;

    const int64_t grid_n = std::max<int64_t>(1, (cols + kBlockN - 1) / kBlockN);
    const int64_t grid_m = std::max<int64_t>(1, (rows + kBlockM - 1) / kBlockM);
    // 64x64 tile 数不足半个 SM 阵列时并行度不够，换 16x16 内核。
    const bool small = grid_n * grid_m * std::max<int64_t>(1, batch) < 32;

    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (small) {
            const dim3 threads(kSmallTile, kSmallTile);
            const dim3 blocks(static_cast<unsigned>((cols + kSmallTile - 1) / kSmallTile),
                              static_cast<unsigned>((rows + kSmallTile - 1) / kSmallTile),
                              static_cast<unsigned>(std::max<int64_t>(1, batch)));
            matmul_small_kernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out),
                rows, cols, k);
        } else {
            const dim3 threads(kThreadsX, kThreadsY);
            const dim3 blocks(static_cast<unsigned>(grid_n), static_cast<unsigned>(grid_m),
                              static_cast<unsigned>(std::max<int64_t>(1, batch)));
            matmul_kernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(out),
                rows, cols, k);
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

    // 半精度操作数先拓宽到 FLOAT32，与 CPU 后端完全一致：16 位格式是存储类型，
    // 不是累加类型。
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
    // 内核读取连续的行主序操作数；dtype 提升走 Typecast（dtype 已匹配时是空
    // 操作）。
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
                                 A.data(), B.data(), dst.data(),
                                 res_type, batch, rows, cols, common),
        "MatMul launch");
    return dst;
}

}  // namespace ops
