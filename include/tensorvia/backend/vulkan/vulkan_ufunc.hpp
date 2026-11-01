#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/tensor_iterator.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops::vk_ufunc {

using ops::iter::broadcast_shapes;
using ops::iter::broadcast_strides;
using ops::iter::same_shape;
using ops::iter::Layout;
using ops::iter::promote;

// 只做"布局 → push constant"的翻译；Vulkan 的 push constant 是 32 位域，所以
// 形状、步长与偏移都要先确认能装下。偏移量随参数走，shader 从 buffer 基址寻址。
// linearizable 布局把 ndim 置 0：shader 据此走连续快路径（两侧 stride 恒为 1），
// 跳过每元素的 stride 解码；shape/strides 留原值，快路径不读它们。
inline BcastVecParams make_bcast_params(const Layout& lay, const Tensor& out) {
    BcastVecParams p{};
    p.ndim  = iter::linearizable(lay) ? 0 : lay.ndim;
    p.numel = lay.numel;
    if (lay.numel > INT32_MAX)
        throw std::runtime_error("vulkan binary op: result exceeds 2^31 elements");
    for (int32_t d = 0; d < lay.ndim; ++d) {
        const int64_t sa = lay.operands[0].strides[d];
        const int64_t sb = lay.operands[1].strides[d];
        if (lay.shape[d] > INT32_MAX || sa > INT32_MAX || sb > INT32_MAX)
            throw std::runtime_error("vulkan binary op: shape/stride exceeds int32");
        p.shape[d] = int32_t(lay.shape[d]);
        p.sa[d]    = int32_t(sa);
        p.sb[d]    = int32_t(sb);
    }
    if (lay.operands[0].offset > INT32_MAX || lay.operands[1].offset > INT32_MAX)
        throw std::runtime_error("vulkan binary op: view offset exceeds int32");
    if (out.offset() > INT32_MAX)
        throw std::runtime_error("vulkan binary op: view offset exceeds int32");
    p.off_a = int32_t(lay.operands[0].offset);
    p.off_b = int32_t(lay.operands[1].offset);
    p.off_o = int32_t(out.offset());
    return p;
}

// 后端策略：把统一迭代层的 Layout 翻译成 push constant 并递交给 shader。
struct VulkanElementwise {
    using Impl = VKTensor;
    using Ctx  = VulkanContext;

    static constexpr Device           device = Device::VULKAN;
    static constexpr std::string_view name   = "vulkan";

    // Vulkan 的提交通道按线程选取，无需显式切换"当前设备"。
    static void activate(Ctx&) {}

    static void launch(Ctx& ctx, const Tensor& A, const Tensor& B, const Tensor& out,
                       const std::shared_ptr<Impl>& ai, const std::shared_ptr<Impl>& bi,
                       const std::shared_ptr<Impl>& oi, DataType, OpType op,
                       const Layout& lay) {
        const BcastVecParams p = make_bcast_params(lay, out);
        ctx.submitCompute(op, A.dtype(), {ai->buffer(), bi->buffer(), oi->buffer()},
                          static_cast<uint32_t>((p.numel + 255) / 256), 1, 1,
                          &p, sizeof(p));
    }
};

inline Tensor bcast_binary(OpType op, const Tensor& A, const Tensor& B,
                           DataType out_dtype, Tensor* dst) {
    return ops::iter::launch_binary<VulkanElementwise>(op, A, B, out_dtype, dst);
}

// 二元算术：dtype 提升 + NumPy 广播。
inline Tensor binary_arith(OpType op, const Tensor& a, const Tensor& b, Tensor* dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, res_type, dst);
}

// 逻辑比较：提升到公共 dtype、比较、写 BOOL。
inline Tensor logical_impl(OpType op, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    // BOOL 没有专用 shader，且其字节可能存着任意非零值，所以 BOOL 操作数经 INT8
    // 比较：cast 会把它们规范化成 0/1。
    const DataType compare_type = res_type == DataType::BOOL ? DataType::INT8 : res_type;
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, compare_type, storage_a);
    const Tensor& B = promote(b, compare_type, storage_b);
    return bcast_binary(op, A, B, DataType::BOOL, nullptr);
}

// 标量比较：右操作数是提升后 dtype 的单元素张量。浮点张量保持自身 dtype；
// 弱类型（整型/BOOL）张量采用默认浮点 dtype。
inline Tensor logical_scalar(OpType op, const Tensor& a, float val) {
    const DataType res_type = scalar_promote_type(a.dtype());
    Tensor storage_a;
    const Tensor& A = promote(a, res_type, storage_a);
    Tensor scalar = ops::Fill(std::vector<int64_t>{1}, res_type, val, Device::VULKAN);
    Tensor storage_b;
    const Tensor& B = promote(scalar, res_type, storage_b);
    return bcast_binary(op, A, B, DataType::BOOL, nullptr);
}

}  // namespace ops::vk_ufunc
