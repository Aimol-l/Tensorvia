// 归约内核——单轴的 sum/min/max/mean、argmax/argmin、all/any 以及 NonZero
// 计数——外加定义相应 Device::CUDA 实现的宿主驱动。
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

// 每个 (outer, inner) 对一个线程，沿被归约的轴行走。
//
// 轴循环的上限按 p.numel 截断：正常布局下恒等（base + axis*inner <= numel 总
// 成立），只有两趟全归约的趟 1 末段会短于 axis。每个输出只多一次 min，不在
// 内层循环里。
template <class T, class Out, class Acc = contract::reduction_acc_t<T>>
__global__ void reduce_axis_kernel(const T* in, Out* out, ReduceAxis p, int op) {
    const int64_t total = p.outer * p.inner;
    const int64_t numel = p.numel > 0 ? p.numel : p.outer * p.axis * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int64_t outer_idx = idx / p.inner;
        const int64_t inner_idx = idx % p.inner;
        const int64_t base = outer_idx * p.axis * p.inner + inner_idx;
        const int64_t limit = min(p.axis, numel - base);

        Acc acc;
        if (op == kReduceSum || op == kReduceMean) {
            acc = Acc(0);
            for (int64_t i = 0; i < limit; ++i) acc += static_cast<Acc>(to_compute<Acc>(in[base + i * p.inner]));
        } else if (op == kReduceProd) {
            acc = Acc(1);
            for (int64_t i = 0; i < limit; ++i) acc *= static_cast<Acc>(to_compute<Acc>(in[base + i * p.inner]));
        } else if (op == kReduceMin) {
            acc = to_compute<Acc>(in[base]);
            for (int64_t i = 1; i < limit; ++i)
                acc = contract::min_contract(acc, to_compute<Acc>(in[base + i * p.inner]));
        } else {
            acc = to_compute<Acc>(in[base]);
            for (int64_t i = 1; i < limit; ++i)
                acc = contract::max_contract(acc, to_compute<Acc>(in[base + i * p.inner]));
        }
        if (op == kReduceMean) {
            const double denom = p.mean_n > 0 ? static_cast<double>(p.mean_n)
                                              : static_cast<double>(p.axis);
            out[idx] = store_as<Out>(static_cast<double>(acc) / denom);
        } else {
            out[idx] = store_as<Out>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// block 协作归约的公共部分。
//
// 线程对输出的划分是"一 block 一个输出、block 内线程沿归约轴分摊"：访存从
// 线程串行整行（每步跨 axis 个元素，同一 warp 命中 32 条不同的 cache line）
// 变成连续合并访问。只有 inner == 1 时行本身连续，所以下面两个内核只走这个
// 几何；inner > 1 的既有"一输出一线程"路径保持不动（那条路径本来就让相邻
// 线程落在相邻元素上）。
// ---------------------------------------------------------------------------

template <class Acc>
TVV_CUDA_HD Acc combine_acc(Acc a, Acc b, int op) {
    if (op == kReduceSum || op == kReduceMean) return a + b;
    if (op == kReduceProd) return a * b;
    if (op == kReduceMin) return contract::min_contract(a, b);
    return contract::max_contract(a, b);
}

// 空集合的单位元。min/max 用有限极值：任何真实元素都会取代它；NaN 仍按
// min_contract/max_contract 的规则传播（与单线程路径一致）。
template <class Acc>
TVV_CUDA_HD Acc identity_acc(int op) {
    if (op == kReduceSum || op == kReduceMean) return Acc(0);
    if (op == kReduceProd) return Acc(1);
    if (op == kReduceMin) return max_finite<Acc>();
    return lowest_finite<Acc>();
}

// 256 个线程的树形归约。blockDim.x 必须是 2 的幂（所有调用点都用 kThreads）。
// 只做设备函数：函数体内的 __shared__ 声明不能出现在宿主编译通道里。
template <class Acc>
__device__ Acc block_reduce_acc(Acc value, int op) {
    __shared__ Acc scratch[kThreads];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            scratch[threadIdx.x] = combine_acc(scratch[threadIdx.x],
                                               scratch[threadIdx.x + offset], op);
        __syncthreads();
    }
    return scratch[0];
}

// inner == 1：一 block 一行。行在输入里连续，block 内线程按线程号跨步读取，
// 同一 warp 的地址相邻。
template <class T, class Out, class Acc = contract::reduction_acc_t<T>>
__global__ void reduce_axis_inner1_kernel(const T* in, Out* out, ReduceAxis p, int op) {
    const int64_t numel = p.numel > 0 ? p.numel : p.outer * p.axis;
    const double denom = p.mean_n > 0 ? static_cast<double>(p.mean_n)
                                      : static_cast<double>(p.axis);
    for (int64_t row = blockIdx.x; row < p.outer; row += gridDim.x) {
        const int64_t base = row * p.axis;
        const int64_t limit = min(p.axis, numel - base);
        Acc acc = identity_acc<Acc>(op);
        for (int64_t i = threadIdx.x; i < limit; i += blockDim.x)
            acc = combine_acc(acc, to_compute<Acc>(in[base + i]), op);
        acc = block_reduce_acc(acc, op);
        if (threadIdx.x == 0) {
            out[row] = (op == kReduceMean)
                           ? store_as<Out>(static_cast<double>(acc) / denom)
                           : store_as<Out>(acc);
        }
        // 下一行会复用同一块 shared，等所有线程读完本轮结果再进入下一轮。
        __syncthreads();
    }
}

// 全归约趟 1：把 [0, numel) 切成 blocks 段，每段连续读取并折叠成一个部分值。
// 与旧的"几何 two-pass"相比，参与的是 blocks * kThreads 个线程（旧路径只有
// blocks 个线程各串行一整段），访存也由每线程跨段读变成 block 内合并读。
template <class T, class Out, class Acc = contract::reduction_acc_t<T>>
__global__ void flat_partial_kernel(const T* in, Out* out, int64_t numel,
                                    int64_t chunk, int op) {
    const int64_t begin = static_cast<int64_t>(blockIdx.x) * chunk;
    const int64_t end = min(begin + chunk, numel);
    Acc acc = identity_acc<Acc>(op);
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        acc = combine_acc(acc, to_compute<Acc>(in[i]), op);
    acc = block_reduce_acc(acc, op);
    if (threadIdx.x == 0) out[blockIdx.x] = store_as<Out>(acc);
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

// All/Any：每个 block 先把自己的片段折叠成一个标志，再用原子 and/or 合并到单个
// 设备字中。宿主先在同一条 stream 上启动 init_kernel，因此目标无需宿主侧清零。
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

// Prod 的累加类型。CPU 后端对非整型一律用 double 累加（reduce_acc_t），而
// reduction_acc_t 在设备端对 32 位浮点仍是 float32。对 1M 量级的连乘，float32
// 的舍入偏差可以到千分点，并且随归约树的形状漂移——同一份输入在"逐行串行"和
// "两趟 block 树"下会给出相差近 1% 的结果。浮点族（含 16 位）的 Prod 因此在
// 设备端也提升到 double 累加：结果不再依赖并行划分方式，也与 CPU 参考只差一次
// 输出舍入。
template <class T>
inline constexpr bool prod_uses_wide_acc =
    std::is_floating_point<T>::value || contract::is_low_precision<T>::value;

// 依 op 与输入类型把累加器标签传给 `f`。Prod 且宽累加时用 double，其余沿用
// 各自 dtype 的 reduction_acc_t。
template <class T, class F>
void with_reduce_acc(int op, F&& f) {
    if constexpr (prod_uses_wide_acc<T>) {
        if (op == kReduceProd) {
            f(type_tag<double>{});
            return;
        }
    }
    (void)op;
    f(type_tag<contract::reduction_acc_t<T>>{});
}

}  // namespace

cudaError_t launch_reduce_axis(cudaStream_t stream, const void* in, void* out,
                               DataType in_dtype, DataType out_dtype, int op,
                               const ReduceAxis& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    const int64_t total = p.outer * p.inner;
    // inner == 1 且轴足够长时走 block 协作路径：一 block 一行，线程沿轴分摊，
    // 读是连续的。轴太短时 block 内树形归约的开销盖过收益，保留旧路径。
    const bool block_path = p.inner == 1 && p.axis >= kAxisInner1MinAxis && p.outer > 0;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using Out = typename decltype(dst_tag)::type;
            const auto launch = [&](auto acc_tag) {
                using Acc = typename decltype(acc_tag)::type;
                if (block_path) {
                    const int64_t rows = p.outer;
                    const int grid = static_cast<int>(
                        rows > std::numeric_limits<int>::max()
                            ? std::numeric_limits<int>::max()
                            : rows);
                    reduce_axis_inner1_kernel<T, Out, Acc><<<grid, kThreads, 0, stream>>>(
                        static_cast<const T*>(in), static_cast<Out*>(out), p, op);
                } else {
                    reduce_axis_kernel<T, Out, Acc><<<blocks_for(total), kThreads, 0, stream>>>(
                        static_cast<const T*>(in), static_cast<Out*>(out), p, op);
                }
            };
            with_reduce_acc<T>(op, launch);
            status = launch_status();
        });
    });
    return status;
}

cudaError_t launch_flat_partial(cudaStream_t stream, const void* in, void* out,
                                int64_t numel, DataType in_dtype, DataType out_dtype,
                                int64_t blocks, int op) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    if (numel <= 0 || blocks <= 0) return cudaErrorInvalidValue;
    if (blocks > std::numeric_limits<int>::max()) blocks = std::numeric_limits<int>::max();
    const int64_t chunk = (numel + blocks - 1) / blocks;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(in_dtype, [&](auto src_tag) {
        using T = typename decltype(src_tag)::type;
        tvv::cuda::dispatch_dtype(out_dtype, [&](auto dst_tag) {
            using Out = typename decltype(dst_tag)::type;
            const auto launch = [&](auto acc_tag) {
                using Acc = typename decltype(acc_tag)::type;
                flat_partial_kernel<T, Out, Acc>
                    <<<static_cast<int>(blocks), kThreads, 0, stream>>>(
                        static_cast<const T*>(in), static_cast<Out*>(out), numel, chunk, op);
            };
            with_reduce_acc<T>(op, launch);
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
// 宿主驱动。
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

// 内核索引扁平行主序 buffer，所以视图在这里一次性物化。传下去的是
// Tensor::data()（而不是 impl 的基址指针），因此带非零 offset 的连续视图读取的是
// 自己的元素。
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
    // 输入的真实元素数：轴循环按它截断（见 reduce_axis_kernel）。正常布局下等于
    // outer*axis*inner，填真值只是让恒等式显式。
    layout.params.numel = static_cast<int64_t>(a.numel());

    for (int i = 0; i < dim; ++i) {
        if (i != axis) layout.shape.push_back(a.shape(i));
    }
// 单轴内核早于 rank-0 张量出现，把空的归约形状写作 {1}；核心层通过
// with_reduced_shape 把结果重塑为真正的 0-dim 张量，所以该回退永远不会到达公开
// API。
    if (layout.shape.empty()) layout.shape = {1};
    return layout;
}

// 全归约的两趟派发。
//
// 轴内核的几何是 outer x axis x inner，全归约展平后是 outer=inner=1、
// axis=numel，于是 total=1：一个 block 里只有 thread 0 在跑，16M 个元素要串行
// 500ms 以上（CPU 只要 2.7ms）。改成先切成 kFlatReduceBlocks 段并行算部分值，
// 再对这 B 个部分值折叠。段长取 ceil，末段短于段长，由内核按 numel 截断。
//
// B 取 1024 足以饱和消费级 GPU；每段 N/1024 个元素（16M 时 64 个/线程）仍是
// 访存受限而非 launch 受限。阈值以下不划算：单线程串行只有数十 µs，多一次派发
// 加一次 scratch 分配回不来。
constexpr int64_t kFlatReduceBlocks   = 1024;
constexpr int64_t kFlatReduceMinNumel = 1 << 16;

bool flat_reduce_applies(const tvv::cuda::ReduceAxis& p) {
    return p.outer * p.inner == 1 && p.axis >= kFlatReduceMinNumel;
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
    const auto stream = static_cast<cudaStream_t>(ctx->stream());
    const tvv::cuda::ReduceAxis& p = layout.params;

    if (flat_reduce_applies(p)) {
        // 块数不超过元素数：极小输入（阈值附近）也得让每段至少有一个元素。
        const int64_t blocks =
            p.axis < kFlatReduceBlocks ? p.axis : kFlatReduceBlocks;
        Tensor partial({blocks}, out_type, Device::CUDA);
        const auto partial_impl = std::dynamic_pointer_cast<CUDATensor>(partial.get_impl());
        if (!partial_impl)
            throw std::runtime_error(std::string("cuda ") + what +
                                     ": failed to allocate the reduction scratch");

        // 趟 1：Mean 先算纯部分和——分母是全归约的元素数，留到趟 2。
        const int pass1_op = (op == tvv::cuda::kReduceMean) ? tvv::cuda::kReduceSum : op;
        tvv::cuda::check_cuda(
            tvv::cuda::launch_flat_partial(stream, src.data(), partial_impl->data(),
                                           p.axis, src.dtype(), out_type, blocks,
                                           pass1_op),
            what);

        // 趟 2：折叠 B 个部分值。输入就是 partial，元素数恰好 B，截断恒等。
        const tvv::cuda::ReduceAxis second{.outer = 1,
                                .axis = blocks,
                                .inner = 1,
                                .numel = blocks,
                                .mean_n = (op == tvv::cuda::kReduceMean) ? p.axis : 0};
        tvv::cuda::check_cuda(
            tvv::cuda::launch_reduce_axis(stream, partial_impl->data(), out_impl->data(),
                                          out_type, out_type, op, second),
            what);
        return out;
    }

    tvv::cuda::check_cuda(
        tvv::cuda::launch_reduce_axis(stream, src.data(), out_impl->data(), src.dtype(),
                                      out_type, op, p),
        what);
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
    return out;
}

bool all_any_value(const Tensor& a, float value, bool is_all, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
// 内核自己初始化这个字（先 init 启动，再原子操作），所以归约前无需宿主往返。
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

    out.to_host();
    return static_cast<const int32_t*>(out.data())[0] != 0;
}

}  // namespace

Tensor SumImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
// Sum 把窄整型与 16 位浮点拓宽到真正的累加类型；所有后端遵循同一条
// reduction_type 规则。
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::cuda::kReduceSum, "Sum");
}

Tensor MinImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::cuda::kReduceMin, "Min");
}

Tensor MaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::cuda::kReduceMax, "Max");
}

Tensor ProdImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    // Prod 与 Sum 同族：窄整型与 16 位浮点拓宽到 reduction_type 再累加。
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::cuda::kReduceProd, "Prod");
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

    out.to_host();
    return static_cast<size_t>(static_cast<const int64_t*>(out.data())[0]);
}

}  // namespace ops
