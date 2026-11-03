// LayerNorm / RmsNorm 的融合内核。
//
// 组合形式（core 的 layer_norm_composed）对每个 8192x4096 的行要跑 8~10 个内核、
// 中间物化多个同尺寸张量；这里每行只读一次输入、写一次输出。行本身（n 个元素）
// 放在 shared 里复用：均值、方差、归一化三趟都从 shared 读，全局只走一次。
// n * sizeof(Acc) 超过默认 shared 上限时退回流式两趟（LN）或一趟（RMS），
// 全局流量增加到 3 读 1 写，但不再需要与行长相等的 shared。
#include "common.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
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
#include "ufunc_detail.hpp"

namespace tvv::cuda {
namespace {

// 单 block 可以静态申请的 shared 上限（sm_89 默认值）。超过它需要
// cudaFuncSetAttribute 显式放开，这里不引入这条宿主依赖。
constexpr int64_t kNormSharedBytes = 48 * 1024;

// 计算类型与逐元素路径一致：整型提升到 float；16 位浮点拓宽到 float；其余保持。
template <class T>
using norm_acc_t =
    std::conditional_t<std::is_integral<T>::value, float, contract::promoted_t<T>>;

// 256 线程的树形求和。结果在最后一次 __syncthreads() 之后写入 scratch[0]，
// 因此所有线程都读到同一个值。两次调用之间由调用方插入 __syncthreads()，
// 避免快线程提前复用 scratch 而慢线程还没取走上一轮结果。
template <class Acc>
__device__ Acc block_sum(Acc value) {
    __shared__ Acc scratch[kThreads];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }
    return scratch[0];
}

// 1 / sqrt(mean + eps)。均值在 Acc 里算，开方与倒数在 double 里做一次，与 CPU
// 内核的 `1.0f / sqrt(var + eps)` 只差一次舍入。
template <class Acc>
__device__ Acc norm_inv_std(Acc mean_square, int64_t n, float eps) {
    const double denom =
        static_cast<double>(mean_square / Acc(n)) + static_cast<double>(eps);
    return static_cast<Acc>(1.0 / ::sqrt(denom));
}

// 行驻留 shared 的版本：每个 block 处理一行，行在 block 内被读三趟（RMS 两趟），
// 全局只有一次加载和一次回写。
template <class T, class Out>
__global__ void norm_shared_kernel(const T* __restrict__ in, const Out* __restrict__ w,
                                   const Out* __restrict__ b, Out* __restrict__ out,
                                   int64_t outer, int64_t n, float eps, int rms) {
    using Acc = norm_acc_t<T>;
    extern __shared__ __align__(16) unsigned char smem[];
    Acc* row = reinterpret_cast<Acc*>(smem);

    for (int64_t r = blockIdx.x; r < outer; r += gridDim.x) {
        const int64_t base = r * n;
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
            row[i] = to_compute<Acc>(in[base + i]);
        __syncthreads();

        Acc mean = Acc(0);
        if (!rms) {
            Acc sum = Acc(0);
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x) sum += row[i];
            sum = block_sum<Acc>(sum);
            __syncthreads();
            mean = sum / Acc(n);
        }

        Acc sq = Acc(0);
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Acc d = rms ? row[i] : row[i] - mean;
            sq += d * d;
        }
        sq = block_sum<Acc>(sq);
        __syncthreads();
        const Acc inv = norm_inv_std<Acc>(sq, n, eps);

        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Acc d = rms ? row[i] : row[i] - mean;
            const Acc y = d * inv * to_compute<Acc>(w[i]);
            out[base + i] =
                rms ? store_as<Out>(y)
                    : store_as<Out>(y + to_compute<Acc>(b[i]));
        }
        // 下一行（若有）会覆盖 row。
        __syncthreads();
    }
}

// 流式版本：行放不进 shared 时，均值/方差/归一化各自扫描一遍全局。LN 是 3 读
// 1 写，RMS 是 2 读 1 写。
template <class T, class Out>
__global__ void norm_stream_kernel(const T* __restrict__ in, const Out* __restrict__ w,
                                   const Out* __restrict__ b, Out* __restrict__ out,
                                   int64_t outer, int64_t n, float eps, int rms) {
    using Acc = norm_acc_t<T>;
    for (int64_t r = blockIdx.x; r < outer; r += gridDim.x) {
        const T* src = in + r * n;
        Out* dst = out + r * n;

        Acc mean = Acc(0);
        if (!rms) {
            Acc sum = Acc(0);
            for (int64_t i = threadIdx.x; i < n; i += blockDim.x)
                sum += to_compute<Acc>(src[i]);
            sum = block_sum<Acc>(sum);
            __syncthreads();
            mean = sum / Acc(n);
        }

        Acc sq = Acc(0);
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Acc v = to_compute<Acc>(src[i]);
            const Acc d = rms ? v : v - mean;
            sq += d * d;
        }
        sq = block_sum<Acc>(sq);
        __syncthreads();
        const Acc inv = norm_inv_std<Acc>(sq, n, eps);

        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            const Acc v = to_compute<Acc>(src[i]);
            const Acc d = rms ? v : v - mean;
            const Acc y = d * inv * to_compute<Acc>(w[i]);
            dst[i] = rms ? store_as<Out>(y)
                         : store_as<Out>(y + to_compute<Acc>(b[i]));
        }
        __syncthreads();
    }
}

}  // namespace

cudaError_t launch_norm(cudaStream_t stream, const void* in, const void* weight,
                        const void* bias, void* out, DataType in_dtype,
                        DataType out_dtype, int64_t outer, int64_t n, float eps,
                        bool rms) {
    if (in == nullptr || weight == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (outer <= 0 || n <= 0) return cudaErrorInvalidValue;
    const int64_t limit = std::numeric_limits<int>::max();
    const int blocks = static_cast<int>(outer > limit ? limit : outer);

    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using Out = typename decltype(dst_tag)::type;
            using Acc = norm_acc_t<T>;
            const int64_t bytes = n * static_cast<int64_t>(sizeof(Acc));
            if (bytes <= kNormSharedBytes) {
                norm_shared_kernel<T, Out><<<blocks, kThreads,
                                             static_cast<size_t>(bytes), stream>>>(
                    static_cast<const T*>(in), static_cast<const Out*>(weight),
                    static_cast<const Out*>(bias), static_cast<Out*>(out), outer, n, eps,
                    rms ? 1 : 0);
            } else {
                norm_stream_kernel<T, Out><<<blocks, kThreads, 0, stream>>>(
                    static_cast<const T*>(in), static_cast<const Out*>(weight),
                    static_cast<const Out*>(bias), static_cast<Out*>(out), outer, n, eps,
                    rms ? 1 : 0);
            }
            status = launch_status();
        });
    });
    return status;
}

}  // namespace tvv::cuda

// ---------------------------------------------------------------------------
// 宿主驱动。
// ---------------------------------------------------------------------------

namespace ops {

namespace {

using ops::cuda_ufunc::stream_of;

// LayerNorm 与 RmsNorm 只差一个仿射偏置和均值项；其余路径共用。
Tensor norm_impl(const Tensor& a, const Tensor* weight, const Tensor* bias, int axis,
                 float eps, bool rms, const char* what) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error(std::string("cuda ") + what + ": axis out of range");

    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= a.shape(i);
    int64_t n = 1;
    for (int i = axis; i < dim; ++i) n *= a.shape(i);

    const DataType out_type =
        is_integral_dtype(a.dtype()) ? DataType::FLOAT32 : a.dtype();

    // 内核按 [outer, n] 的连续行主序解释输入；跨步视图先物化。
    const Tensor src = a.is_contiguous() ? a : a.clone();
    const Tensor w = weight->is_contiguous() ? *weight : weight->clone();
    const bool bias_owned = bias != nullptr && !bias->is_contiguous();
    const Tensor b_storage = bias_owned ? bias->clone() : Tensor();
    const Tensor* b = bias_owned ? &b_storage : bias;

    const auto a_shape = a.shape();
    Tensor out(std::vector<int64_t>(a_shape.begin(), a_shape.end()), out_type,
               Device::CUDA);

    auto ctx = cuda_ufunc::context_of(src, what);
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(out.get_impl());
    if (!out_impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": failed to allocate the result");
    if (!std::dynamic_pointer_cast<CUDATensor>(w.get_impl()))
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": weight does not live on the CUDA device");
    if (b != nullptr && !std::dynamic_pointer_cast<CUDATensor>(b->get_impl()))
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": bias does not live on the CUDA device");

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_norm(stream_of(*ctx), src.data(), w.data(),
                               b != nullptr ? b->data() : nullptr, out.data(),
                               src.dtype(), out_type, outer, n, eps, rms),
        what);
    return out;
}

}  // namespace

Tensor LayerNormImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& weight,
                                            const Tensor& bias, int axis, float eps) {
    return norm_impl(a, &weight, &bias, axis, eps, false, "LayerNorm");
}

Tensor RmsNormImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& weight,
                                          int axis, float eps) {
    return norm_impl(a, &weight, nullptr, axis, eps, true, "RmsNorm");
}

}  // namespace ops
