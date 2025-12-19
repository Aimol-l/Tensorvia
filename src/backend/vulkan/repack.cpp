#include "backend/vulkan/ops/repack.h"
#include "backend/vulkan/vulkan_constant.h"


void RepackImpl<Device::VULKAN>::execute(
    const Metadata& meta,
    vk::Buffer input,
    vk::Buffer output,
    std::shared_ptr<VulkanContext> ctx)
{
    RepackParams params{};
    params.ndim       = meta.shape.size();
    params.dtype_size = calc_dtype_size(meta.dtype);
    params.numel      = meta.numel;
    for (int i = 0; i < 6; ++i) {
        params.shape[i]   = (i < meta.shape.size())   ? meta.shape[i]   : 1;
        params.strides[i] = (i < meta.strides.size()) ? meta.strides[i] : 0;
    }
    uint32_t gx = (meta.numel + 255) / 256;
    ctx->submitCompute(
        OpType::Repack,
        DataType::INT8,
        {input,output},
        gx, 1, 1,
        &params, sizeof(params)
    );
}

template struct RepackImpl<Device::VULKAN>;
