// sort / topk 内核与 Device::CUDA 宿主驱动。
//
// 排序键是 (值, 原始下标) 对：下标做次级键，于是任何排序算法都天然稳定——
// 平局时更靠前的下标在前，与 CPU 的 std::stable_sort 语义逐位一致（这也是
// topk 的 largest/smallest 在平局上的次序）。
//
// 两条路径，按轴长分派：
//   n <= 32    一个线程负责一列，私有内存插入排序（每列 <= 32 对，L1 驻留）；
//   n <= 1024  一个 block 负责一列，共享内存 bitonic 网络（补齐到 2 的幂，
//              尾部哨兵排在最后，只写前 keep 个）；
// 轴长超过 1024 显式抛错——共享内存装不下，也不静默给错结果。
//
// 内核索引扁平行主序 buffer，所以非连续视图在宿主侧一次性物化；传下去的是
// Tensor::data()（已含视图 offset），带 offset 的连续视图读的是自己的元素。
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

// 轴长上限：block 路径的共享内存按 padded * sizeof(Pair<T>) 申请，double 的
// 1024 对是 16KB，仍在默认 48KB 静态/动态共享预算内。
inline constexpr int kSmallAxis = 32;
inline constexpr int kMaxSortAxis = 1024;

// (值, 下标) 对在共享内存里的布局。int32 下标够用：轴长上限 1024，且内核只在
// 轴内编号；写出时再拓宽成 INT64。
template <class T>
struct Pair {
    T v;
    int32_t i;
};

// 全序：先比值；相等比下标（稳定）。NaN 与 CPU 的 `<` 比较一样两边皆为假，
// 位置不定——核心测试不用 NaN 喂排序。
template <class T>
__device__ inline bool pair_before(T va, int32_t ia, T vb, int32_t ib, bool desc) {
    if (va == vb) return ia < ib;
    return desc ? (vb < va) : (va < vb);
}

// 路径 A：一个线程一列，插入排序。
template <class T>
__global__ void sort_small_kernel(const T* in, T* out_vals, int64_t* out_idx,
                                  SortAxis p) {
    const int64_t columns = p.outer * p.inner;
    const int64_t col     = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= columns) return;
    const int64_t o = col / p.inner;
    const int64_t c = col % p.inner;
    const T* base = in + o * p.n * p.inner + c;

    T vals[kSmallAxis];
    int32_t idx[kSmallAxis];
    const int nn = static_cast<int>(p.n);
    for (int i = 0; i < nn; ++i) {
        vals[i] = base[static_cast<int64_t>(i) * p.inner];
        idx[i]  = static_cast<int32_t>(i);
    }
    for (int i = 1; i < nn; ++i) {
        const T v = vals[i];
        const int32_t x = idx[i];
        int j = i - 1;
        while (j >= 0 && pair_before(v, x, vals[j], idx[j], p.descending != 0)) {
            vals[j + 1] = vals[j];
            idx[j + 1]  = idx[j];
            --j;
        }
        vals[j + 1] = v;
        idx[j + 1]  = x;
    }

    T* dv      = out_vals + o * p.keep * p.inner + c;
    int64_t* di = out_idx + o * p.keep * p.inner + c;
    const int keep = static_cast<int>(p.keep);
    for (int i = 0; i < keep; ++i) {
        dv[static_cast<int64_t>(i) * p.inner] = vals[i];
        di[static_cast<int64_t>(i) * p.inner] = idx[i];
    }
}

// 路径 B：一个 block 一列，共享内存 bitonic。compare-exchange 用 pair_before
// 作全序，段方向由 (i & k) 决定——对任意全序都产出按该序排列的结果。
template <class T>
__global__ void sort_block_kernel(const T* in, T* out_vals, int64_t* out_idx,
                                  SortAxis p) {
    extern __shared__ char smem_raw[];
    Pair<T>* s = reinterpret_cast<Pair<T>*>(smem_raw);
    const int P = static_cast<int>(p.padded);

    const int64_t col = blockIdx.x;
    const int64_t o = col / p.inner;
    const int64_t c = col % p.inner;
    const T* base = in + o * p.n * p.inner + c;
    const bool desc = p.descending != 0;

    // 装载 + 哨兵补齐。哨兵值排在所在次序的尾部：升序取最大值，降序取最小值；
    // 下标取 INT32_MAX，与真实元素同值时也排在后面。
    const T sentinel = desc ? lowest_finite<T>() : max_finite<T>();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        if (i < static_cast<int>(p.n)) {
            s[i].v = base[static_cast<int64_t>(i) * p.inner];
            s[i].i = static_cast<int32_t>(i);
        } else {
            s[i].v = sentinel;
            // INT32_MAX 而非 std::numeric_limits<int32_t>::max()：后者是宿主
            // constexpr 函数，nvcc 拒绝从 __global__ 调用（见 common.cuh 的
            // max_finite/lowest_finite 注释）。宏与语义一致，<cstdint> 已包含。
            s[i].i = INT32_MAX;
        }
    }
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < P; i += blockDim.x) {
                const int l = i ^ j;
                if (l <= i) continue;
                const Pair<T> a = s[i];
                const Pair<T> b = s[l];
                bool do_swap;
                if ((i & k) == 0) {
                    do_swap = pair_before(b.v, b.i, a.v, a.i, desc);  // 升序段
                } else {
                    do_swap = pair_before(a.v, a.i, b.v, b.i, desc);  // 降序段
                }
                if (do_swap) {
                    s[i] = b;
                    s[l] = a;
                }
            }
            __syncthreads();
        }
    }

    T* dv      = out_vals + o * p.keep * p.inner + c;
    int64_t* di = out_idx + o * p.keep * p.inner + c;
    const int keep = static_cast<int>(p.keep);
    for (int i = threadIdx.x; i < keep; i += blockDim.x) {
        dv[static_cast<int64_t>(i) * p.inner] = s[i].v;
        di[static_cast<int64_t>(i) * p.inner] = s[i].i;
    }
}

}  // namespace

cudaError_t launch_sort(cudaStream_t stream, const void* in, void* out_vals,
                        void* out_idx, DataType dtype, const SortAxis& p) {
    if (in == nullptr || out_vals == nullptr || out_idx == nullptr)
        return cudaErrorInvalidValue;
    if (p.n <= 0 || p.keep <= 0 || p.outer * p.inner <= 0) return cudaSuccess;
    const int64_t columns = p.outer * p.inner;
    cudaError_t status = cudaErrorNotSupported;
    tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (p.n <= kSmallAxis) {
            sort_small_kernel<T><<<static_cast<int>(columns), kThreads, 0, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out_vals),
                static_cast<int64_t*>(out_idx), p);
        } else {
            const int shared = static_cast<int>(p.padded * sizeof(Pair<T>));
            sort_block_kernel<T><<<static_cast<int>(columns), kThreads, shared, stream>>>(
                static_cast<const T*>(in), static_cast<T*>(out_vals),
                static_cast<int64_t*>(out_idx), p);
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

std::shared_ptr<CUDAContext> sort_context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<CUDATensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("cuda ") + what +
                                 ": operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("cuda ") + what + ": missing CUDAContext");
    return ctx;
}

Tensor sort_materialize(const Tensor& a) {
    return a.is_contiguous() ? a : a.clone();
}

struct SortGeometry {
    tvv::cuda::SortAxis params{};
    std::vector<int64_t> shape;  // 输出形状（dim 维已换成 keep）
};

SortGeometry sort_geometry(const Tensor& in, int dim, int64_t keep, bool descending,
                           const char* what) {
    const int rank = static_cast<int>(in.shape().size());
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank)
        throw std::runtime_error(std::string("cuda ") + what + ": axis out of range");

    SortGeometry g;
    g.params.outer = 1;
    g.params.n     = in.shape(static_cast<size_t>(dim));
    g.params.inner = 1;
    for (int i = 0; i < dim; ++i) g.params.outer *= in.shape(static_cast<size_t>(i));
    for (size_t i = static_cast<size_t>(dim) + 1; i < in.shape().size(); ++i)
        g.params.inner *= in.shape(i);
    if (keep < 0) keep = g.params.n;  // 全排序
    if (keep > g.params.n)
        throw std::runtime_error(std::format(
            "topk: k ({}) must be between 1 and the axis length ({})", keep, g.params.n));
    g.params.keep       = keep;
    g.params.descending = descending ? 1 : 0;
    if (g.params.n > tvv::cuda::kMaxSortAxis)
        throw std::runtime_error(std::format(
            "cuda {}: axis length {} exceeds the device sort limit ({})",
            what, g.params.n, tvv::cuda::kMaxSortAxis));

    int64_t padded = 1;
    while (padded < g.params.n) padded <<= 1;
    g.params.padded = padded;

    g.shape = std::vector<int64_t>(in.shape().begin(), in.shape().end());
    g.shape[static_cast<size_t>(dim)] = keep;
    return g;
}

// sort 与 topk 共用的驱动：排好每列后取前 keep 个。keep < 0 表示全排序。
ops::SortResult run_sort(const Tensor& input, int dim, int64_t keep, bool descending,
                         const char* what) {
    const Tensor src = sort_materialize(input);
    auto ctx = sort_context_of(src, what);
    const SortGeometry g = sort_geometry(src, dim, keep, descending, what);

    Tensor values(g.shape, src.dtype(), Device::CUDA);
    Tensor indices(g.shape, DataType::INT64, Device::CUDA);
    const auto vals_impl = std::dynamic_pointer_cast<CUDATensor>(values.get_impl());
    const auto idx_impl  = std::dynamic_pointer_cast<CUDATensor>(indices.get_impl());
    if (!vals_impl || !idx_impl)
        throw std::runtime_error(std::string("cuda ") + what + ": failed to allocate results");
    // 空几何（空轴、keep 为 0、没有列）没有可启动的内核，输出就是空张量。
    if (g.params.n <= 0 || g.params.keep <= 0 || g.params.outer * g.params.inner <= 0)
        return ops::SortResult{std::move(values), std::move(indices)};

    ctx->activate();
    tvv::cuda::check_cuda(
        tvv::cuda::launch_sort(static_cast<cudaStream_t>(ctx->stream()), src.data(),
                               vals_impl->data(), idx_impl->data(), src.dtype(), g.params),
        what);
    return ops::SortResult{std::move(values), std::move(indices)};
}

}  // namespace

SortResult SortImpl<Device::CUDA>::execute(const Tensor& input, int dim, bool descending) {
    return run_sort(input, dim, -1, descending, "Sort");
}

TopkResult TopkImpl<Device::CUDA>::execute(const Tensor& input, int64_t k, int dim,
                                           bool largest) {
    // topk 就是"排好序后取前 k 个"：largest 对应降序。k 的合法性（>=1 且 <= 轴长）
    // 由核心层先把守，这里直接排。
    ops::SortResult sorted = run_sort(input, dim, k, largest, "Topk");
    return TopkResult{std::move(sorted.values), std::move(sorted.indices)};
}

}  // namespace ops
