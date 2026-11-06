#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

Tensor conv_pool_execute(const Tensor& input, const Tensor& weight, const Tensor* bias,
                         const ConvPoolShape& shape) {
    vulkan_detail::require_int32_indexable(shape.numel(), "conv_pool");

    Tensor dst({shape.conv.batch, shape.conv.out_ch, shape.out_h, shape.out_w},
               input.dtype(), Device::VULKAN);

    auto in_impl  = std::dynamic_pointer_cast<VKTensor>(input.get_impl());
    auto w_impl   = std::dynamic_pointer_cast<VKTensor>(weight.get_impl());
    auto out_impl = std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    if (!in_impl || !w_impl || !out_impl)
        throw std::runtime_error("vulkan conv_pool: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(in_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan conv_pool: missing VulkanContext");

    // 描述符布局要求 binding 2 非空。没有偏置时着色器不会读它（has_bias == 0），
    // 所以直接绑权重 buffer 当占位，不为一次调用去分配一块零向量。
    vk::Buffer bias_buffer = w_impl->buffer();
    if (bias != nullptr) {
        auto b_impl = std::dynamic_pointer_cast<VKTensor>(bias->get_impl());
        if (!b_impl)
            throw std::runtime_error("vulkan conv_pool: bias is not a VKTensor");
        bias_buffer = b_impl->buffer();
    }

    const ConvPoolParams params{.batch    = static_cast<uint32_t>(shape.conv.batch),
                                .in_ch   = static_cast<uint32_t>(shape.conv.in_ch),
                                .in_h    = static_cast<uint32_t>(shape.conv.in_h),
                                .in_w    = static_cast<uint32_t>(shape.conv.in_w),
                                .out_ch  = static_cast<uint32_t>(shape.conv.out_ch),
                                .kh      = static_cast<uint32_t>(shape.conv.kh),
                                .kw      = static_cast<uint32_t>(shape.conv.kw),
                                .stride  = static_cast<uint32_t>(shape.conv.stride),
                                .padding = static_cast<uint32_t>(shape.conv.padding),
                                .conv_h  = static_cast<uint32_t>(shape.conv.out_h),
                                .conv_w  = static_cast<uint32_t>(shape.conv.out_w),
                                .out_h   = static_cast<uint32_t>(shape.out_h),
                                .out_w   = static_cast<uint32_t>(shape.out_w),
                                .has_bias = static_cast<uint32_t>(bias != nullptr ? 1 : 0)};

    // 与 shader/conv_pool/*.comp 的几何一致：一 invocation 一输出，256 个
    // 工作项一个 workgroup，一维 grid。
    constexpr uint32_t kWorkgroup = 256;
    const uint32_t gx = static_cast<uint32_t>((shape.numel() + kWorkgroup - 1) / kWorkgroup);
    ctx_impl->submitCompute(OpType::ConvPool, input.dtype(),
                            {in_impl->buffer(), w_impl->buffer(), bias_buffer,
                             out_impl->buffer()},
                            gx, 1, 1, &params, sizeof(params));
    return dst;
}

}  // namespace

Tensor Conv2dReluMaxPool2dImpl<Device::VULKAN>::execute(const Tensor& input,
                                                        const Tensor& weight,
                                                        const Tensor* bias,
                                                        const ConvPoolShape& shape) {
    return conv_pool_execute(input, weight, bias, shape);
}

}  // namespace ops
