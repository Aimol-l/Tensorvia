#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

// DataType -> shader 里的 dst_type 编码。编码顺序见 shader/typecast/*.comp 与
// vulkan_common.hpp 的 kCastDst* 常量。
int32_t cast_dst_code(DataType t) {
    switch (t) {
        case DataType::BOOL:     return kCastDstBool;
        case DataType::INT8:     return kCastDstInt8;
        case DataType::INT16:    return kCastDstInt16;
        case DataType::INT32:    return kCastDstInt32;
        case DataType::INT64:    return kCastDstInt64;
        case DataType::FLOAT16:  return kCastDstFloat16;
        case DataType::BFLOAT16: return kCastDstBFloat16;
        case DataType::FLOAT32:  return kCastDstFloat32;
        case DataType::FLOAT64:  return kCastDstFloat64;
    }
    throw std::runtime_error("vulkan Typecast: unknown destination dtype");
}

// 源 buffer -> 目标 buffer 的一次类型转换。
//
// 目标 buffer 被重复绑定到 9 个不同标量类型的别名上（见 shader 的注释与
// kTypecastBindings），由 push constant 选中实际写入的那一个；未选中的别名在
// shader 里不会被访问。Vulkan 允许描述符别名。
void submit_cast(const Tensor& src, const VKTensor& dst, DataType dst_type) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(src.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Typecast: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Typecast: missing VulkanContext");
    if (src.numel() == 0) return;

    std::vector<vk::Buffer> buffers;
    buffers.reserve(kTypecastBindings);
    buffers.push_back(src_impl->buffer());
    for (int i = 1; i < kTypecastBindings; ++i) buffers.push_back(dst.buffer());

    TypecastParams params{.numel    = static_cast<int64_t>(src.numel()),
                          .dst_type = cast_dst_code(dst_type)};
    const uint32_t gx = static_cast<uint32_t>((src.numel() + 255) / 256);
    // 源是 BOOL 时 submitCompute 把流水线键换成 int8：bool buffer 里存的就是
    // 规范化后的 0/1，按 int8 读与按 bool 读等价。
    ctx_impl->submitCompute(OpType::Typecast, src.dtype(), buffers, gx, 1, 1, &params,
                            sizeof(params));
}

}  // namespace

Tensor TypecastImpl<Device::VULKAN>::execute(const Tensor& a, DataType dst_type) {
    Tensor result(a.shape(), dst_type, a.device());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!dst_impl) throw std::runtime_error("vulkan Typecast: expected a VKTensor");
    submit_cast(a, *dst_impl, dst_type);
    return result;
}

void TypecastImpl<Device::VULKAN>::execute(Tensor& a, DataType dst_type) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Typecast: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Typecast: missing VulkanContext");

    // 元素宽度变了，旧 buffer 装不下：换一块新 buffer，再让旧的在途提交结束后
    // 退役（VKTensor 析构走 context 的退役队列）。
    auto dst_impl = std::make_shared<VKTensor>(a.numel(), dst_type, ctx_impl);
    submit_cast(a, *dst_impl, dst_type);
    a.set_impl(dst_impl, dst_type);
}

}  // namespace ops
