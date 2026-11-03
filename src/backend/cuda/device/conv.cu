// 2-D 卷积的设备实现：一个线程算一个输出元素。
//
// 输出是连续 NCHW，所以线程的扁平下标直接解码成 (n, oc, oy, ox)，不需要任何
// 跨线程协作——卷积的每个输出都独立。权重与输入全程线性寻址；累加器取提升后的
// 计算类型，半精度输入不会在半精度里累积误差。
#include "common.cuh"

#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;
using tvv::cuda::contract::promoted_t;

// 分块几何（两个 tiled 内核共用）。
constexpr int kConvTile       = 32;  // 输出 tile 的边长
constexpr int kConvThreads    = 64;  // 8x8 = 64
constexpr int kConvThreadTile = 4;   // 每线程 4x4 输出
constexpr int kConvThreadsX   = kConvTile / kConvThreadTile;  // 8
constexpr int kConvThreadsY   = kConvTile / kConvThreadTile;  // 8
constexpr int kConvMaxKernel  = 5;   // kh/kw 上限（patch 最大 36x36）
constexpr int kConvMaxOcTile  = 4;   // 单 block 的输出通道数上限

// 3x3 专用版本：寄存器缓存每个线程的 6x6 输入块。
//
// 通用版本在通道循环里每做一次 FMA 就要读一次 shared，访存:计算 = 1:1，shared
// 吞吐成为上限。这里每个通道只把 (4+3-1)^2 = 36 个输入装进寄存器，随后
// 4 个输出通道 x 9 个 tap x 16 个输出 = 576 次 FMA 全部吃寄存器，比例变成
// 16:1。核尺寸在编译期固定，展开也完整发生。
template <class T>
__global__ void conv2d_tiled3_kernel(const T* __restrict__ in, const T* __restrict__ weight,
                                     const T* __restrict__ bias, T* __restrict__ out,
                                     ops::Conv2dShape p) {
    using Acc = tvv::cuda::contract::promoted_t<T>;
    constexpr int oc_tile = sizeof(Acc) >= 8 ? 2 : kConvMaxOcTile;
    constexpr int kSide   = kConvThreadTile + 3 - 1;  // 6

    __shared__ T patch[kConvTile + 2][kConvTile + 2];
    __shared__ T wsh[oc_tile * 9];

    const int64_t groups = (p.out_ch + oc_tile - 1) / oc_tile;
    const int64_t g   = blockIdx.z % groups;
    const int64_t n   = blockIdx.z / groups;
    const int64_t oc0 = g * oc_tile;
    const int64_t oy0 = static_cast<int64_t>(blockIdx.y) * kConvTile;
    const int64_t ox0 = static_cast<int64_t>(blockIdx.x) * kConvTile;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int64_t in_plane = p.in_h * p.in_w;

    Acc acc[oc_tile][kConvThreadTile][kConvThreadTile];
#pragma unroll
    for (int o = 0; o < oc_tile; ++o) {
        const int64_t oc = oc0 + o;
        const Acc init = (bias != nullptr && oc < p.out_ch)
                             ? tvv::cuda::to_compute<Acc>(bias[oc])
                             : Acc(0);
#pragma unroll
        for (int i = 0; i < kConvThreadTile; ++i)
#pragma unroll
            for (int j = 0; j < kConvThreadTile; ++j) acc[o][i][j] = init;
    }

    for (int64_t c = 0; c < p.in_ch; ++c) {
        const T* in_c = in + (n * p.in_ch + c) * in_plane;
        for (int64_t idx = tid; idx < (kConvTile + 2) * (kConvTile + 2);
             idx += kConvThreads) {
            const int64_t py = idx / (kConvTile + 2);
            const int64_t px = idx % (kConvTile + 2);
            const int64_t iy = oy0 - p.padding + py;
            const int64_t ix = ox0 - p.padding + px;
            patch[py][px] = (iy >= 0 && iy < p.in_h && ix >= 0 && ix < p.in_w)
                                ? in_c[iy * p.in_w + ix]
                                : T(0);
        }
        for (int64_t idx = tid; idx < oc_tile * 9; idx += kConvThreads) {
            const int64_t o  = idx / 9;
            const int64_t r  = idx % 9;
            const int64_t oc = oc0 + o;
            wsh[idx] = oc < p.out_ch ? weight[(oc * p.in_ch + c) * 9 + r] : T(0);
        }
        __syncthreads();

        Acc rp[kSide][kSide];
#pragma unroll
        for (int i = 0; i < kSide; ++i)
#pragma unroll
            for (int j = 0; j < kSide; ++j)
                rp[i][j] = tvv::cuda::to_compute<Acc>(
                    patch[ty * kConvThreadTile + i][tx * kConvThreadTile + j]);

#pragma unroll
        for (int o = 0; o < oc_tile; ++o) {
#pragma unroll
            for (int i = 0; i < 3; ++i) {
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const Acc wv = tvv::cuda::to_compute<Acc>(wsh[o * 9 + i * 3 + j]);
#pragma unroll
                    for (int di = 0; di < kConvThreadTile; ++di) {
#pragma unroll
                        for (int dj = 0; dj < kConvThreadTile; ++dj)
                            acc[o][di][dj] += rp[di + i][dj + j] * wv;
                    }
                }
            }
        }
        __syncthreads();
    }

    const int64_t out_plane = p.out_h * p.out_w;
#pragma unroll
    for (int o = 0; o < oc_tile; ++o) {
        const int64_t oc = oc0 + o;
        if (oc >= p.out_ch) continue;
        T* out_c = out + (n * p.out_ch + oc) * out_plane;
#pragma unroll
        for (int di = 0; di < kConvThreadTile; ++di) {
            const int64_t oy = oy0 + ty * kConvThreadTile + di;
            if (oy >= p.out_h) continue;
#pragma unroll
            for (int dj = 0; dj < kConvThreadTile; ++dj) {
                const int64_t ox = ox0 + tx * kConvThreadTile + dj;
                if (ox >= p.out_w) continue;
                out_c[oy * p.out_w + ox] = tvv::cuda::store_as<T>(acc[o][di][dj]);
            }
        }
    }
}

// 分块卷积：一个 block 算 (n, oc 组) 的一个 32x32 输出 tile，每个线程算其中
// 4x4，同时算 kConvMaxOcTile 个输出通道。
//
// 旧内核一输出一线程：每个输出都要沿 C x kh x kw 重新从全局/缓存读 576 个
// 输入，线程之间没有显式复用。现在每处理一个输入通道，整个 block 先把该通道
// 覆盖到的 (32+kh-1) x (32+kw-1) 输入 patch 收进 shared，之后 block 内所有输出
// 都从 shared 读。再让一个 block 顺带算多个输出通道：同一份 patch 被 4 个通道
// 复用，输入侧全局流量按通道组数等比下降——56x56x64 的基准形状下这是主要瓶颈。
template <class T>
__global__ void conv2d_tiled_kernel(const T* __restrict__ in, const T* __restrict__ weight,
                                    const T* __restrict__ bias, T* __restrict__ out,
                                    ops::Conv2dShape p) {
    using Acc = tvv::cuda::contract::promoted_t<T>;

    // 一个通道的 4x4 累加器是 16 个寄存器；double 下两个通道就把预算吃满，
    // float 与 16 位类型可以放 4 个。
    constexpr int oc_tile = sizeof(Acc) >= 8 ? 2 : kConvMaxOcTile;

    __shared__ T patch[kConvTile + kConvMaxKernel - 1][kConvTile + kConvMaxKernel - 1];
    __shared__ T wsh[oc_tile * kConvMaxKernel * kConvMaxKernel];

    const int64_t groups = (p.out_ch + oc_tile - 1) / oc_tile;
    const int64_t g  = blockIdx.z % groups;
    const int64_t n  = blockIdx.z / groups;
    const int64_t oc0 = g * oc_tile;
    const int64_t oy0 = static_cast<int64_t>(blockIdx.y) * kConvTile;
    const int64_t ox0 = static_cast<int64_t>(blockIdx.x) * kConvTile;

    const int tx = threadIdx.x;  // 0..7，负责 4 列
    const int ty = threadIdx.y;  // 0..7，负责 4 行
    const int tid = ty * blockDim.x + tx;

    const int64_t patch_h = kConvTile + p.kh - 1;
    const int64_t patch_w = kConvTile + p.kw - 1;
    const int64_t in_plane  = p.in_h * p.in_w;
    const int64_t taps      = p.kh * p.kw;

    Acc acc[oc_tile][kConvThreadTile][kConvThreadTile];
#pragma unroll
    for (int o = 0; o < oc_tile; ++o) {
        const int64_t oc = oc0 + o;
        const Acc init = (bias != nullptr && oc < p.out_ch)
                             ? tvv::cuda::to_compute<Acc>(bias[oc])
                             : Acc(0);
#pragma unroll
        for (int i = 0; i < kConvThreadTile; ++i)
#pragma unroll
            for (int j = 0; j < kConvThreadTile; ++j) acc[o][i][j] = init;
    }

    for (int64_t c = 0; c < p.in_ch; ++c) {
        const T* in_c = in + (n * p.in_ch + c) * in_plane;
        for (int64_t idx = tid; idx < patch_h * patch_w; idx += kConvThreads) {
            const int64_t py = idx / patch_w;
            const int64_t px = idx % patch_w;
            const int64_t iy = oy0 - p.padding + py;
            const int64_t ix = ox0 - p.padding + px;
            patch[py][px] = (iy >= 0 && iy < p.in_h && ix >= 0 && ix < p.in_w)
                                ? in_c[iy * p.in_w + ix]
                                : T(0);
        }
        // 权重同样每通道只装一次；每组 9 个值在整个 block 内广播。
        for (int64_t idx = tid; idx < oc_tile * taps; idx += kConvThreads) {
            const int64_t o  = idx / taps;
            const int64_t r  = idx % taps;
            const int64_t oc = oc0 + o;
            wsh[idx] = oc < p.out_ch
                           ? weight[oc * p.in_ch * taps + c * taps + r]
                           : T(0);
        }
        __syncthreads();

#pragma unroll
        for (int o = 0; o < oc_tile; ++o) {
            const T* w_c = wsh + o * taps;
            for (int64_t i = 0; i < p.kh; ++i) {
                for (int64_t j = 0; j < p.kw; ++j) {
                    const Acc wv = tvv::cuda::to_compute<Acc>(w_c[i * p.kw + j]);
#pragma unroll
                    for (int di = 0; di < kConvThreadTile; ++di) {
#pragma unroll
                        for (int dj = 0; dj < kConvThreadTile; ++dj) {
                            acc[o][di][dj] +=
                                tvv::cuda::to_compute<Acc>(
                                    patch[ty * kConvThreadTile + di + i]
                                         [tx * kConvThreadTile + dj + j]) *
                                wv;
                        }
                    }
                }
            }
        }
        // 下一通道会覆盖 patch 与 wsh。
        __syncthreads();
    }

    const int64_t out_plane = p.out_h * p.out_w;
#pragma unroll
    for (int o = 0; o < oc_tile; ++o) {
        const int64_t oc = oc0 + o;
        if (oc >= p.out_ch) continue;
        T* out_c = out + (n * p.out_ch + oc) * out_plane;
#pragma unroll
        for (int di = 0; di < kConvThreadTile; ++di) {
            const int64_t oy = oy0 + ty * kConvThreadTile + di;
            if (oy >= p.out_h) continue;
#pragma unroll
            for (int dj = 0; dj < kConvThreadTile; ++dj) {
                const int64_t ox = ox0 + tx * kConvThreadTile + dj;
                if (ox >= p.out_w) continue;
                out_c[oy * p.out_w + ox] = tvv::cuda::store_as<T>(acc[o][di][dj]);
            }
        }
    }
}

template <class T>
__global__ void conv2d_kernel(const T* in, const T* weight, const T* bias, T* out,
                              ops::Conv2dShape p) {
    // 不用 Conv2dShape::numel()：它是宿主成员函数，设备代码里调不到。
    const int64_t total = p.batch * p.out_ch * p.out_h * p.out_w;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t ox   = idx % p.out_w;
    int64_t rest       = idx / p.out_w;
    const int64_t oy   = rest % p.out_h;
    rest /= p.out_h;
    const int64_t oc   = rest % p.out_ch;
    const int64_t n    = rest / p.out_ch;

    using Acc = promoted_t<T>;
    Acc acc = bias != nullptr ? tvv::cuda::to_compute<Acc>(bias[oc]) : Acc(0);

    const int64_t in_plane  = p.in_h * p.in_w;
    const int64_t out_plane = p.out_h * p.out_w;
    const int64_t taps      = p.in_ch * p.kh * p.kw;

    for (int64_t c = 0; c < p.in_ch; ++c) {
        const T* in_c = in + (n * p.in_ch + c) * in_plane;
        const T* w_c  = weight + oc * taps + c * p.kh * p.kw;
        for (int64_t i = 0; i < p.kh; ++i) {
            const int64_t y = oy * p.stride + i - p.padding;
            if (y < 0 || y >= p.in_h) continue;
            const T* in_row = in_c + y * p.in_w;
            const T* w_row  = w_c + i * p.kw;
            for (int64_t j = 0; j < p.kw; ++j) {
                const int64_t x = ox * p.stride + j - p.padding;
                if (x < 0 || x >= p.in_w) continue;
                acc += tvv::cuda::to_compute<Acc>(in_row[x]) *
                       tvv::cuda::to_compute<Acc>(w_row[j]);
            }
        }
    }

    out[(n * p.out_ch + oc) * out_plane + oy * p.out_w + ox] = tvv::cuda::store_as<T>(acc);
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_conv2d(cudaStream_t stream, const void* in, const void* weight,
                          const void* bias, void* out, DataType dtype,
                          const ops::Conv2dShape& p) {
    if (in == nullptr || weight == nullptr || out == nullptr) return cudaErrorInvalidValue;
    // tiled 路径要求 stride == 1（patch 的坐标按步长 1 推导）、核大小不超过
    // shared patch 的静态上限。(n, oc 组) 必须放进 blockIdx.z——每 block 的通道
    // 组数随标量类型变化，所以这半条判定放到 dtype 派发之后。
    const bool tiled_geom = p.stride == 1 && p.kh > 0 && p.kw > 0 &&
                            p.kh <= kConvMaxKernel && p.kw <= kConvMaxKernel &&
                            p.in_ch > 0 && p.out_ch > 0;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        constexpr int oc_tile = sizeof(tvv::cuda::contract::promoted_t<T>) >= 8 ? 2
                                                                                : kConvMaxOcTile;
        const int64_t groups = (p.out_ch + oc_tile - 1) / oc_tile;
        const int64_t planes = p.batch * groups;
        if (tiled_geom && planes > 0 && planes <= 65535) {
            const dim3 threads(kConvThreadsX, kConvThreadsY);
            const dim3 grid(static_cast<unsigned>((p.out_w + kConvTile - 1) / kConvTile),
                            static_cast<unsigned>((p.out_h + kConvTile - 1) / kConvTile),
                            static_cast<unsigned>(planes));
            if (p.kh == 3 && p.kw == 3) {
                conv2d_tiled3_kernel<T><<<grid, threads, 0, stream>>>(
                    static_cast<const T*>(in), static_cast<const T*>(weight),
                    static_cast<const T*>(bias), static_cast<T*>(out), p);
            } else {
                conv2d_tiled_kernel<T><<<grid, threads, 0, stream>>>(
                    static_cast<const T*>(in), static_cast<const T*>(weight),
                    static_cast<const T*>(bias), static_cast<T*>(out), p);
            }
        } else {
            conv2d_kernel<T><<<blocks_for(p.numel()), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<const T*>(weight),
                static_cast<const T*>(bias), static_cast<T*>(out), p);
        }
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace ops {

Tensor Conv2dImpl<Device::CUDA>::execute(const Tensor& input, const Tensor& weight,
                                         const Tensor* bias, const Conv2dShape& shape) {
    using namespace ops::cuda_ufunc;

    // 公共入口已经把操作数整理成连续布局；这里只兜住绕过入口的调用。
    const Tensor src = input.is_contiguous() ? input : input.clone();
    const Tensor w   = weight.is_contiguous() ? weight : weight.clone();

    Tensor result({shape.batch, shape.out_ch, shape.out_h, shape.out_w}, src.dtype(),
                  Device::CUDA);

    const auto in_impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!in_impl) throw std::runtime_error("cuda conv2d: operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(in_impl->context());
    if (!ctx) throw std::runtime_error("cuda conv2d: missing CUDAContext");

    const auto w_impl = std::dynamic_pointer_cast<CUDATensor>(w.get_impl());
    if (!w_impl) throw std::runtime_error("cuda conv2d: weight does not live on the CUDA device");

    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!out_impl) throw std::runtime_error("cuda conv2d: failed to allocate the result");

    const void* bias_ptr = nullptr;
    if (bias != nullptr) {
        const auto b_impl = std::dynamic_pointer_cast<CUDATensor>(bias->get_impl());
        if (!b_impl) throw std::runtime_error("cuda conv2d: bias does not live on the CUDA device");
        bias_ptr = b_impl->data();
    }

    ctx->activate();
    tvv::cuda::check_cuda(::tvv::cuda::launch_conv2d(stream_of(*ctx), in_impl->data(),
                                                    w_impl->data(), bias_ptr, out_impl->data(),
                                                    src.dtype(), shape),
                          "conv2d launch");
    return result;
}

}  // namespace ops
