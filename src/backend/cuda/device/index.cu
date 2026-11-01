// 条件选择、索引与累积和：where / index_select / gather / cumsum 的设备实现。
//
// cond 与 index 的标量类型不参与内核模板实例化：它们在核心里按 dtype 序号查表
// 读取。否则 where 会是 9×9、index_select/gather 会是 9×2 个实例，光编译时间就
// 抵得上整个后端——而分支的条件是 warp 内一致的，代价只是每个元素一次跳转。
#include "common.cuh"

#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

#include "cuda_launch.hpp"
#include "ufunc_detail.hpp"

namespace {

using tvv::cuda::blocks_for;
using tvv::cuda::kThreads;
using tvv::cuda::launch_status;
using tvv::cuda::type_tag;

// 扁平输出下标 -> 三侧元素偏移。广播维的步长为 0，偏移自带视图起点。
TVV_CUDA_HD int64_t bcast_offset3(int64_t idx, const tvv::cuda::Select& p,
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

// 非零即真，与 CPU 的 `cond[i] != C(0)` 一致（NaN 因此为真）。
TVV_CUDA_HD bool cond_truth(const void* p, int64_t i, int dtype) {
    switch (dtype) {
        case static_cast<int>(DataType::BOOL):
            return static_cast<const uint8_t*>(p)[i] != 0;
        case static_cast<int>(DataType::INT8):
            return static_cast<const int8_t*>(p)[i] != 0;
        case static_cast<int>(DataType::INT16):
            return static_cast<const int16_t*>(p)[i] != 0;
        case static_cast<int>(DataType::INT32):
            return static_cast<const int32_t*>(p)[i] != 0;
        case static_cast<int>(DataType::INT64):
            return static_cast<const int64_t*>(p)[i] != 0;
        case static_cast<int>(DataType::BFLOAT16):
            return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]) != 0.0f;
        case static_cast<int>(DataType::FLOAT16):
            return __half2float(static_cast<const __half*>(p)[i]) != 0.0f;
        case static_cast<int>(DataType::FLOAT32):
            return static_cast<const float*>(p)[i] != 0.0f;
        case static_cast<int>(DataType::FLOAT64):
            return static_cast<const double*>(p)[i] != 0.0;
        default:
            return false;
    }
}

// 索引只有 INT32/INT64 两种（公开入口已经拒绝其它 dtype）。
TVV_CUDA_HD int64_t load_index(const void* p, int64_t i, int dtype) {
    if (dtype == static_cast<int>(DataType::INT64))
        return static_cast<const int64_t*>(p)[i];
    return static_cast<const int32_t*>(p)[i];
}

// 越界索引截断到 0：调用方会读标志并抛异常，但即使没读，内核也不能越界读写。
TVV_CUDA_HD int64_t clamp_index(int64_t v, int64_t axis_len, int32_t* flag) {
    if (v < 0 || v >= axis_len) {
        if (flag != nullptr) *flag = 1;
        return 0;
    }
    return v;
}

// 写出时把 BOOL 规范化为 0/1：源的字节可能是 2 之类的非规范值。
template <class T>
TVV_CUDA_HD T normalize_store(T v) {
    if constexpr (std::is_same_v<T, uint8_t>) return v != 0 ? uint8_t{1} : uint8_t{0};
    else return v;
}

// ---------------------------------------------------------------------------
// 内核。
// ---------------------------------------------------------------------------

template <class T>
__global__ void select_kernel(const void* cond, const T* a, const T* b, T* out,
                              tvv::cuda::Select p) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        const bool pick_a = cond_truth(cond, p.off_c + bcast_offset3(i, p, p.sc), p.cond_dtype);
        const int64_t ia = p.off_a + bcast_offset3(i, p, p.sa);
        const int64_t ib = p.off_b + bcast_offset3(i, p, p.sb);
        out[p.off_o + i] = normalize_store(pick_a ? a[ia] : b[ib]);
    }
}

// 输出元素 (o, k, inn) = 输入 (o, idx[k], inn)；按输出扁平下标一元素一线程。
template <class T>
__global__ void index_select_kernel(const T* in, const void* idx, T* out,
                                    tvv::cuda::IndexAxis p, int idx_dtype,
                                    int32_t* flag) {
    const int64_t total = p.outer * p.idx_len * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const int64_t inn = i % p.inner;
        const int64_t t   = i / p.inner;
        const int64_t k   = t % p.idx_len;
        const int64_t o   = t / p.idx_len;
        const int64_t r   = clamp_index(load_index(idx, k, idx_dtype), p.axis_len, flag);
        out[i] = in[(o * p.axis_len + r) * p.inner + inn];
    }
}

template <class T>
__global__ void gather_kernel(const T* in, const void* idx, T* out,
                              tvv::cuda::Gather p, int32_t* flag) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < p.numel; i += stride) {
        // 扁平下标 -> 坐标；非 dim 维的坐标按输入步长累加成 base。
        int64_t base = 0;
        int64_t rem  = i;
        for (int d = p.ndim - 1; d >= 0; --d) {
            const int64_t dim = p.out_shape[d];
            if (dim <= 0) continue;
            const int64_t coord = rem % dim;
            rem /= dim;
            if (d != p.dim) base += coord * p.in_strides[d];
        }
        const int64_t r = clamp_index(load_index(idx, i, p.idx_dtype), p.axis_dim, flag);
        out[i] = in[base + r * p.axis_stride];
    }
}

// 一个线程负责一条 (outer, inner) 柱；累积顺序与 CPU 内核完全一致。
template <class T>
__global__ void cumsum_kernel(const T* in, T* out, tvv::cuda::IndexAxis p) {
    const int64_t total  = p.outer * p.inner;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < total; task += stride) {
        const int64_t o   = task / p.inner;
        const int64_t inn = task % p.inner;
        const T* src = in + o * p.axis_len * p.inner + inn;
        T* dst       = out + o * p.axis_len * p.inner + inn;
        T acc = T(0);
        for (int64_t k = 0; k < p.axis_len; ++k) {
            acc += src[k * p.inner];
            dst[k * p.inner] = acc;
        }
    }
}

}  // namespace

namespace tvv::cuda {

cudaError_t launch_select(cudaStream_t stream, const void* cond, const void* a,
                          const void* b, void* out, DataType val_dtype,
                          const Select& p) {
    if (cond == nullptr || a == nullptr || b == nullptr || out == nullptr)
        return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(val_dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        select_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            cond, static_cast<const T*>(a), static_cast<const T*>(b),
            static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_index_select(cudaStream_t stream, const void* in, const void* idx,
                                void* out, DataType val_dtype, DataType idx_dtype,
                                const IndexAxis& p, int32_t* flag) {
    if (in == nullptr || idx == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(val_dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        index_select_kernel<T><<<blocks_for(p.outer * p.idx_len * p.inner), kThreads, 0,
                                 stream>>>(static_cast<const T*>(in), idx,
                                           static_cast<T*>(out), p,
                                           static_cast<int>(idx_dtype), flag);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_gather(cudaStream_t stream, const void* in, const void* idx,
                          void* out, DataType val_dtype, DataType idx_dtype,
                          const Gather& p, int32_t* flag) {
    if (in == nullptr || idx == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(val_dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        gather_kernel<T><<<blocks_for(p.numel), kThreads, 0, stream>>>(
            static_cast<const T*>(in), idx, static_cast<T*>(out), p, flag);
        status = launch_status();
    });
    return status;
}

cudaError_t launch_cumsum(cudaStream_t stream, const void* in, void* out,
                          DataType dtype, const IndexAxis& p) {
    if (in == nullptr || out == nullptr) return cudaErrorInvalidValue;
    cudaError_t status = cudaErrorNotSupported;
    ::tvv::cuda::dispatch_dtype(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        cumsum_kernel<T><<<blocks_for(p.outer * p.inner), kThreads, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), p);
        status = launch_status();
    });
    return status;
}

}  // namespace tvv::cuda

namespace {

using ops::cuda_ufunc::context_of;
using ops::cuda_ufunc::stream_of;

// 内核线性遍历连续布局，所以跨步视图先一次物化。传下去的是 Tensor::data()
//（已含视图偏移）。
Tensor materialize(const Tensor& t) { return t.is_contiguous() ? t : t.clone(); }

// 索引越界的契约来自 CPU：越界即刻抛 "index out of range"。设备端没有异常，
// 所以用一个小内核置标志，再把 4 字节读回宿主。这一次同步是这类算子唯一需要宿主
// 参与的地方（NonZero 读计数同理），换来的是三个后端同一份错误行为。
void require_indices_in_range(Tensor flag, const char* op) {
    flag.to_host();
    if (static_cast<const int32_t*>(flag.data())[0] != 0)
        throw std::runtime_error(std::string(op) + ": index out of range");
}

}  // namespace

namespace ops {

Tensor WhereImpl<Device::CUDA>::execute(const Tensor& cond, const Tensor& a,
                                        const Tensor& b) {
    using namespace ops::cuda_ufunc;

    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor pa, pb;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }

    const Tensor src = materialize(cond);
    const Tensor Ac  = materialize(*A);
    const Tensor Bc  = materialize(*B);

    const iter::PreparedInputs in({&src, &Ac, &Bc}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::CUDA);

    const auto impl   = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    const auto a_impl = std::dynamic_pointer_cast<CUDATensor>(Ac.get_impl());
    const auto b_impl = std::dynamic_pointer_cast<CUDATensor>(Bc.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!impl || !a_impl || !b_impl || !out_impl)
        throw std::runtime_error("cuda where: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error("cuda where: missing CUDAContext");

    require_dims(static_cast<size_t>(lay.ndim), "where");
    ::tvv::cuda::Select p{};
    p.ndim       = lay.ndim;
    p.numel      = lay.numel;
    p.cond_dtype = static_cast<int>(in[0].dtype());
    for (int32_t d = 0; d < lay.ndim; ++d) {
        p.shape[d] = lay.shape[d];
        p.sc[d]    = lay.operands[0].strides[d];
        p.sa[d]    = lay.operands[1].strides[d];
        p.sb[d]    = lay.operands[2].strides[d];
    }
    p.off_c = lay.operands[0].offset;
    p.off_a = lay.operands[1].offset;
    p.off_b = lay.operands[2].offset;
    p.off_o = result.offset();

    ctx->activate();
    // 偏移量随 Select 一起传入，所以这里给 impl 基址——用 Tensor::data() 会把视图
    // offset 叠加两次。
    ::tvv::cuda::check_cuda(
        ::tvv::cuda::launch_select(stream_of(*ctx), impl->data(), a_impl->data(),
                                   b_impl->data(), out_impl->data(), res_type, p),
        "where launch");
    return result;
}

// 标量版本：把标量铺成 0 维张量，与张量-张量路径共用同一个内核。用 0 维而不是
// {1}：两者对非空广播结果等价，但全 0 维输入的广播结果必须仍是 0 维（CPU 如此）。
Tensor WhereImpl<Device::CUDA>::execute(const Tensor& cond, const Tensor& a, float b) {
    const DataType res_type = scalar_promote_type(a.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, b, Device::CUDA);
    return WhereImpl<Device::CUDA>::execute(cond, a, scalar);
}

Tensor WhereImpl<Device::CUDA>::execute(const Tensor& cond, float a, const Tensor& b) {
    const DataType res_type = scalar_promote_type(b.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, a, Device::CUDA);
    return WhereImpl<Device::CUDA>::execute(cond, scalar, b);
}

Tensor IndexSelectImpl<Device::CUDA>::execute(const Tensor& input, int dim,
                                              const Tensor& index) {
    using namespace ops::cuda_ufunc;

    const Tensor src = materialize(input);
    const Tensor idx = materialize(index);

    const auto in_shape = src.shape();
    const int64_t idx_len = static_cast<int64_t>(idx.numel());

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end());
    out_shape[static_cast<size_t>(dim)] = idx_len;
    Tensor result(out_shape, input.dtype(), Device::CUDA);

    ::tvv::cuda::IndexAxis p{};
    p.axis_len = in_shape[static_cast<size_t>(dim)];
    p.idx_len  = idx_len;
    p.inner    = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < in_shape.size(); ++i)
        p.inner *= in_shape[i];
    p.outer = 1;
    for (int i = 0; i < dim; ++i) p.outer *= in_shape[static_cast<size_t>(i)];

    const auto impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    const auto idx_impl = std::dynamic_pointer_cast<CUDATensor>(idx.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!impl || !idx_impl || !out_impl)
        throw std::runtime_error("cuda index_select: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error("cuda index_select: missing CUDAContext");

    // 标志必须显式清零：Tensor 的裸构造不会初始化设备内存。
    Tensor flag = ops::Zeros(std::vector<int64_t>{1}, DataType::INT32, Device::CUDA);
    const auto flag_impl = std::dynamic_pointer_cast<CUDATensor>(flag.get_impl());

    ctx->activate();
    ::tvv::cuda::check_cuda(
        ::tvv::cuda::launch_index_select(stream_of(*ctx), impl->data(), idx_impl->data(),
                                         out_impl->data(), src.dtype(), idx.dtype(), p,
                                         static_cast<int32_t*>(flag_impl->data())),
        "index_select launch");
    require_indices_in_range(flag, "index_select");
    return result;
}

Tensor GatherImpl<Device::CUDA>::execute(const Tensor& input, int dim,
                                         const Tensor& index) {
    using namespace ops::cuda_ufunc;

    const Tensor src = materialize(input);
    const Tensor idx = materialize(index);

    const auto in_shape = src.shape();
    const auto idx_shape = idx.shape();
    const size_t ndim = in_shape.size();

    Tensor result(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()), input.dtype(),
                  Device::CUDA);

    require_dims(ndim, "gather");
    ::tvv::cuda::Gather p{};
    p.ndim  = static_cast<int32_t>(ndim);
    p.numel = static_cast<int64_t>(result.numel());
    p.dim   = dim;
    p.axis_dim = in_shape[static_cast<size_t>(dim)];
    // 输入连续，步长就地求；axis_stride 单独留一份给越界判定的邻居。
    int64_t acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        p.in_strides[d] = acc;
        acc *= in_shape[d];
    }
    p.axis_stride = p.in_strides[static_cast<size_t>(dim)];
    for (size_t d = 0; d < ndim; ++d) p.out_shape[d] = idx_shape[d];

    const auto impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    const auto idx_impl = std::dynamic_pointer_cast<CUDATensor>(idx.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!impl || !idx_impl || !out_impl)
        throw std::runtime_error("cuda gather: operands must live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error("cuda gather: missing CUDAContext");

    // 标志必须显式清零：Tensor 的裸构造不会初始化设备内存。
    Tensor flag = ops::Zeros(std::vector<int64_t>{1}, DataType::INT32, Device::CUDA);
    const auto flag_impl = std::dynamic_pointer_cast<CUDATensor>(flag.get_impl());

    ctx->activate();
    ::tvv::cuda::check_cuda(
        ::tvv::cuda::launch_gather(stream_of(*ctx), impl->data(), idx_impl->data(),
                                   out_impl->data(), src.dtype(), idx.dtype(), p,
                                   static_cast<int32_t*>(flag_impl->data())),
        "gather launch");
    require_indices_in_range(flag, "gather");
    return result;
}

Tensor CumsumImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    using namespace ops::cuda_ufunc;

    const Tensor src = materialize(a);
    Tensor result(a.shape(), a.dtype(), Device::CUDA);

    ::tvv::cuda::IndexAxis p{};
    const auto shape = src.shape();
    p.axis_len = shape[static_cast<size_t>(axis)];
    p.inner    = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i)
        p.inner *= shape[i];
    p.outer = 1;
    for (int i = 0; i < axis; ++i) p.outer *= shape[static_cast<size_t>(i)];

    const auto impl = std::dynamic_pointer_cast<CUDATensor>(src.get_impl());
    if (!impl) throw std::runtime_error("cuda cumsum: operand does not live on the CUDA device");
    auto ctx = std::dynamic_pointer_cast<CUDAContext>(impl->context());
    if (!ctx) throw std::runtime_error("cuda cumsum: missing CUDAContext");
    const auto out_impl = std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    if (!out_impl) throw std::runtime_error("cuda cumsum: failed to allocate the result");

    ctx->activate();
    ::tvv::cuda::check_cuda(
        ::tvv::cuda::launch_cumsum(stream_of(*ctx), impl->data(), out_impl->data(),
                                   src.dtype(), p),
        "cumsum launch");
    return result;
}

}  // namespace ops
