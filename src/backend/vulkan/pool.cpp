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

// 与 shader/{max,avg}_pool2d/*.comp 的几何一致：一 invocation 一输出，
// 256 个工作项一个 workgroup，grid 是一维的。
Tensor pool2d_execute(const Tensor& input, const Pool2dShape& shape, bool average) {
    vulkan_detail::require_int32_indexable(shape.numel(), "pool2d");

    Tensor dst({shape.batch, shape.channels, shape.out_h, shape.out_w}, input.dtype(),
               Device::VULKAN);

    auto in_impl  = std::dynamic_pointer_cast<VKTensor>(input.get_impl());
    auto out_impl = std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    if (!in_impl || !out_impl)
        throw std::runtime_error("vulkan pool2d: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(in_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan pool2d: missing VulkanContext");

    const Pool2dParams params{.batch    = static_cast<uint32_t>(shape.batch),
                              .channels = static_cast<uint32_t>(shape.channels),
                              .in_h     = static_cast<uint32_t>(shape.in_h),
                              .in_w     = static_cast<uint32_t>(shape.in_w),
                              .kh       = static_cast<uint32_t>(shape.kh),
                              .kw       = static_cast<uint32_t>(shape.kw),
                              .stride_h = static_cast<uint32_t>(shape.stride_h),
                              .stride_w = static_cast<uint32_t>(shape.stride_w),
                              .padding  = static_cast<uint32_t>(shape.padding),
                              .out_h    = static_cast<uint32_t>(shape.out_h),
                              .out_w    = static_cast<uint32_t>(shape.out_w)};

    constexpr uint32_t kWorkgroup = 256;
    const uint32_t gx = static_cast<uint32_t>((shape.numel() + kWorkgroup - 1) / kWorkgroup);
    ctx_impl->submitCompute(average ? OpType::AvgPool2d : OpType::MaxPool2d, input.dtype(),
                            {in_impl->buffer(), out_impl->buffer()},
                            gx, 1, 1, &params, sizeof(params));
    return dst;
}

}  // namespace

Tensor MaxPool2dImpl<Device::VULKAN>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, false);
}

Tensor AvgPool2dImpl<Device::VULKAN>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, true);
}

}  // namespace ops
