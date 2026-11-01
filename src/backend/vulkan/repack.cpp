#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"

void vulkan_repack(const Metadata& meta, vk::Buffer input, vk::Buffer output,
                   std::shared_ptr<VulkanContext> ctx) {
    if (!ctx) throw std::runtime_error("vulkan_repack: null VulkanContext");
    RepackParams params{};
    params.ndim       = static_cast<uint32_t>(meta.shape.size());
    params.dtype_size = static_cast<uint32_t>(calc_dtype_size(meta.dtype));
    params.numel      = static_cast<uint32_t>(meta.numel);
    // 视图起点：没有它，切片视图会从 buffer 基址重排，形状/步长都对，值却全错。
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
    // 绑定顺序必须与 repack_int8.comp 一致：绑定 0 是 Dst（写），绑定 1 是 Src
    // （读）。
    ctx->submitCompute(OpType::Repack, DataType::INT8, {output, input}, gx, 1, 1, &params,
                       sizeof(params));
}
