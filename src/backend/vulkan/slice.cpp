#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/vulkan/vulkan_ops.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

Tensor SliceImpl<Device::VULKAN>::execute(
    const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const int ndim       = static_cast<int>(t.shape().size());
    const int slice_dims = static_cast<int>(ranges.size());

    if (slice_dims > ndim) throw std::runtime_error("Slice: invalid ranges");

    // ---------- output shape ----------
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim);
    for (int i = 0; i < slice_dims; ++i)
        out_shape.push_back(ranges[i].second - ranges[i].first);
    for (int i = slice_dims; i < ndim; ++i) out_shape.push_back(t.shape(i));

    Tensor res(out_shape, t.dtype(), Device::VULKAN);

    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    if (!res_impl || !src_impl) throw std::runtime_error("vulkan Slice: expected VKTensors");
    auto ctx = std::dynamic_pointer_cast<VulkanContext>(res_impl->context());
    if (!ctx) throw std::runtime_error("vulkan Slice: missing VulkanContext");

    // ---------- fill params ----------
    SliceParams p{};
    p.ndim       = static_cast<uint32_t>(ndim);
    p.slice_dims = static_cast<uint32_t>(slice_dims);

    auto in_shape   = t.shape();
    auto out_shape2 = res.shape();
    auto in_strides = t.strides();

    // Source view origin: without it, slicing a view reads from the buffer
    // base instead of the view start.
    if (t.offset() > INT32_MAX)
        throw std::runtime_error(
            "slice (Vulkan): view offset exceeds the int32 range of the kernel");
    p.offset = static_cast<uint32_t>(t.offset());

    for (int i = 0; i < 6; ++i) {
        const size_t slot    = static_cast<size_t>(i);
        p.input_shape[i]   = (slot < in_shape.size())   ? static_cast<uint32_t>(in_shape[i])   : 1;
        p.output_shape[i]  = (slot < out_shape2.size()) ? static_cast<uint32_t>(out_shape2[i]) : 1;
        p.input_strides[i] = (slot < in_strides.size()) ? static_cast<uint32_t>(in_strides[i]) : 0;
        p.slice_starts[i]  =
            (i < slice_dims) ? static_cast<uint32_t>(ranges[i].first) : 0;
    }

    const uint32_t gx = static_cast<uint32_t>((res.numel() + 255) / 256);
    ctx->submitCompute(OpType::Slice, t.dtype(), {src_impl->buffer(), res_impl->buffer()}, gx,
                       1, 1, &p, sizeof(p));
    return res;
}

}  // namespace ops
