// Device::VULKAN 的 where / index_select / gather / cumsum。
//
// 几何形状与 CPU 实现逐项对齐：输入视为 outer x axis x inner，gather 用输出扁平
// 下标解码坐标。跨步视图在这里物化成连续布局，设备内核只做线性寻址。
//
// 两个与其它后端不同的地方，都源于 Vulkan 只认识固定 dtype 的着色器：
//   * where 的 cond 在宿主上转成 FLOAT64（本后端的 Typecast 是宿主往返）。着色器
//     只判 "!= 0"，而任何 dtype 到 double 的转换都不会把非零变成零，NaN 仍是真，
//     所以判真结果与 CPU 的 `cond[i] != C(0)` 完全一致。
//   * index 不需要转换：缓冲按 32 位字读，`idx_is_64` 决定一个下标占几个字。
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/tensor_iterator.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

std::shared_ptr<VKTensor> impl_of(const Tensor& t, const char* what) {
    auto impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
    return impl;
}

std::shared_ptr<VulkanContext> context_of(const Tensor& t, const char* what) {
    auto ctx = std::dynamic_pointer_cast<VulkanContext>(impl_of(t, what)->context());
    if (!ctx)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    return ctx;
}

// 内核线性遍历连续布局，所以跨步视图先一次物化。
Tensor materialize(const Tensor& t) { return t.is_contiguous() ? t : t.clone(); }

// 索引越界的契约来自 CPU：越界即刻抛 "index out of range"。设备端没有异常，所以
// 用一个标志 + 4 字节回读代替。这一次同步是这类算子唯一需要宿主参与的地方
//（NonZero 读计数同理），换来的是四个后端同一份错误行为。
Tensor make_flag() {
    return ops::Zeros(std::vector<int64_t>{1}, DataType::INT32, Device::VULKAN);
}

void require_indices_in_range(Tensor& flag, const char* op) {
    flag.to_host();
    if (static_cast<const int32_t*>(flag.data())[0] != 0)
        throw std::runtime_error(std::string(op) + ": index out of range");
}

uint32_t grid_of(int64_t numel) { return static_cast<uint32_t>((numel + 255) / 256); }

// 本后端的着色器用 32 位元素下标，几何量因此都要装进 int32。张量的元素总数已经
// 在分配时被限制在 INT32_MAX 内，这里检查的是由形状推导出来的量（outer/inner 等
// 都是 numel 的因子，天然满足），所以失败只可能来自调用方传进来的非法形状。
int32_t to_dim(int64_t v, const char* what) {
    if (v < 0 || v > INT32_MAX)
        throw std::runtime_error(std::string("vulkan ") + what + ": dimension exceeds int32");
    return static_cast<int32_t>(v);
}

}  // namespace

Tensor WhereImpl<Device::VULKAN>::execute(const Tensor& cond, const Tensor& a,
                                          const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor pa, pb;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }

    // 着色器把 cond 读成 double，因此这里先把它带到 FLOAT64。FLOAT64 但非连续的
    // 视图仍要物化：内核从 buffer 基址线性寻址。
    Tensor cond64;
    const Tensor* C = &cond;
    if (cond.dtype() != DataType::FLOAT64) {
        cond64 = ops::Typecast(cond, DataType::FLOAT64);
        C      = &cond64;
    } else if (!cond.is_contiguous()) {
        cond64 = cond.clone();
        C      = &cond64;
    }

    const iter::PreparedInputs in({C, A, B}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::VULKAN);

    WhereParams p{};
    p.numel = lay.numel;
    p.ndim  = lay.ndim;
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

    auto ctx        = context_of(result, "where");
    auto params_buf = ctx->createBuffer<WhereParams>(p);
    ctx->submitCompute(OpType::Where, res_type,
                       {params_buf.buffer(), impl_of(*C, "where")->buffer(),
                        impl_of(*A, "where")->buffer(), impl_of(*B, "where")->buffer(),
                        impl_of(result, "where")->buffer()},
                       grid_of(lay.numel), 1, 1, nullptr, 0);
    return result;
}

// 标量版本：把标量铺成 0 维张量，与张量-张量路径共用同一个内核。用 0 维而不是
// {1}：全 0 维输入的广播结果必须仍是 0 维（CPU 如此）。
Tensor WhereImpl<Device::VULKAN>::execute(const Tensor& cond, const Tensor& a, float b) {
    const DataType res_type = scalar_promote_type(a.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, b, Device::VULKAN);
    return WhereImpl<Device::VULKAN>::execute(cond, a, scalar);
}

Tensor WhereImpl<Device::VULKAN>::execute(const Tensor& cond, float a, const Tensor& b) {
    const DataType res_type = scalar_promote_type(b.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, a, Device::VULKAN);
    return WhereImpl<Device::VULKAN>::execute(cond, scalar, b);
}

Tensor IndexSelectImpl<Device::VULKAN>::execute(const Tensor& input, int dim,
                                                const Tensor& index) {
    const Tensor src = materialize(input);
    const Tensor idx = materialize(index);

    const auto in_shape   = src.shape();
    const int64_t idx_len = static_cast<int64_t>(idx.numel());

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end());
    out_shape[static_cast<size_t>(dim)] = idx_len;
    Tensor result(out_shape, input.dtype(), Device::VULKAN);

    IndexAxisParams p{};
    p.axis_len = to_dim(in_shape[static_cast<size_t>(dim)], "index_select");
    p.idx_len  = to_dim(idx_len, "index_select");
    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < in_shape.size(); ++i)
        inner *= in_shape[i];
    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= in_shape[static_cast<size_t>(i)];
    p.inner     = to_dim(inner, "index_select");
    p.outer     = to_dim(outer, "index_select");
    p.idx_is_64 = idx.dtype() == DataType::INT64 ? 1 : 0;

    Tensor flag     = make_flag();
    const int64_t n = static_cast<int64_t>(p.outer) * p.idx_len * p.inner;
    context_of(result, "index_select")
        ->submitCompute(OpType::IndexSelect, src.dtype(),
                        {impl_of(src, "index_select")->buffer(),
                         impl_of(idx, "index_select")->buffer(),
                         impl_of(result, "index_select")->buffer(),
                         impl_of(flag, "index_select")->buffer()},
                        grid_of(n), 1, 1, &p, sizeof(p));
    require_indices_in_range(flag, "index_select");
    return result;
}

Tensor GatherImpl<Device::VULKAN>::execute(const Tensor& input, int dim,
                                           const Tensor& index) {
    const Tensor src     = materialize(input);
    const Tensor idx     = materialize(index);

    const auto in_shape  = src.shape();
    const auto idx_shape = idx.shape();
    const size_t ndim    = in_shape.size();

    std::vector<int64_t> out_shape_v(idx_shape.begin(), idx_shape.end());
    Tensor result(out_shape_v, input.dtype(), Device::VULKAN);

    GatherParams p{};
    p.numel     = to_dim(static_cast<int64_t>(result.numel()), "gather");
    p.ndim      = static_cast<int32_t>(ndim);
    p.dim       = dim;
    p.idx_is_64 = idx.dtype() == DataType::INT64 ? 1 : 0;
    p.axis_dim  = to_dim(in_shape[static_cast<size_t>(dim)], "gather");

    // 输入已物化成连续布局，非 dim 维的步长因此由形状直接推出。
    int64_t acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        p.in_strides[d] = to_dim(acc, "gather");
        acc *= in_shape[d];
    }
    p.axis_stride = p.in_strides[static_cast<size_t>(dim)];
    for (size_t d = 0; d < ndim; ++d) p.out_shape[d] = to_dim(idx_shape[d], "gather");

    Tensor flag = make_flag();
    context_of(result, "gather")
        ->submitCompute(OpType::Gather, src.dtype(),
                        {impl_of(src, "gather")->buffer(),
                         impl_of(idx, "gather")->buffer(),
                         impl_of(result, "gather")->buffer(),
                         impl_of(flag, "gather")->buffer()},
                        grid_of(p.numel), 1, 1, &p, sizeof(p));
    require_indices_in_range(flag, "gather");
    return result;
}

Tensor CumsumImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    const Tensor src = materialize(a);
    Tensor result(a.shape(), a.dtype(), Device::VULKAN);

    const auto shape = src.shape();
    CumsumParams p{};
    p.axis_len = to_dim(shape[static_cast<size_t>(axis)], "cumsum");
    int64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= shape[i];
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[static_cast<size_t>(i)];
    p.inner = to_dim(inner, "cumsum");
    p.outer = to_dim(outer, "cumsum");

    // 一个线程一条 (outer, inner) 柱，派发网格按柱数算，不按元素数。
    const int64_t columns = static_cast<int64_t>(p.outer) * p.inner;
    context_of(result, "cumsum")
        ->submitCompute(OpType::Cumsum, src.dtype(),
                        {impl_of(src, "cumsum")->buffer(), impl_of(result, "cumsum")->buffer()},
                        grid_of(columns), 1, 1, &p, sizeof(p));
    return result;
}

}  // namespace ops
