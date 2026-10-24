#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"

void vulkan_repack(const Metadata& meta, vk::Buffer input, vk::Buffer output,
                   std::shared_ptr<VulkanContext> ctx) {
    if (!ctx) throw std::runtime_error("vulkan_repack: null VulkanContext");
    RepackParams params{};
    params.ndim       = static_cast<uint32_t>(meta.shape.size());
    params.dtype_size = static_cast<uint32_t>(calc_dtype_size(meta.dtype));
    params.numel      = static_cast<uint32_t>(meta.numel);
    // View origin: without it, a sliced view is repacked from the buffer base
    // with correct shape/strides but wholly wrong values.
    params.offset = static_cast<uint32_t>(meta.offset);
    for (int i = 0; i < 6; ++i) {
        const size_t slot = static_cast<size_t>(i);
        params.shape[i]   = (slot < meta.shape.size())
                                ? static_cast<uint32_t>(meta.shape[i])
                                : 1;
        params.strides[i] = (slot < meta.strides.size())
                                ? static_cast<uint32_t>(meta.strides[i])
                                : 0;
    }
    const uint32_t gx = static_cast<uint32_t>((meta.numel + 255) / 256);
    // Binding order must match repack_int8.comp: binding 0 is Dst (write),
    // binding 1 is Src (read).
    ctx->submitCompute(OpType::Repack, DataType::INT8, {output, input}, gx, 1, 1, &params,
                       sizeof(params));
}
