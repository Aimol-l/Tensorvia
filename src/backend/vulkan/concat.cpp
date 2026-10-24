#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/vulkan/vulkan_ops.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

Tensor ConcatImpl<Device::VULKAN>::execute(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::runtime_error("Concat: empty inputs");
    if (tensors.size() > 8) throw std::runtime_error("Concat: max 8 inputs");

    const int ndim = static_cast<int>(tensors[0].shape().size());
    if (dim < 0 || dim >= ndim) throw std::runtime_error("Concat: invalid axis");
    const DataType dtype = tensors[0].dtype();

    // ---------- output shape ----------
    DimArray out_shape;
    out_shape.assign(tensors[0].shape().begin(), tensors[0].shape().end());
    out_shape[dim] = 0;
    for (auto& t : tensors) out_shape[dim] += t.shape(dim);

    Tensor res(out_shape, dtype, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    if (!res_impl) throw std::runtime_error("vulkan Concat: expected a VKTensor");
    auto ctx = std::dynamic_pointer_cast<VulkanContext>(res_impl->context());
    if (!ctx) throw std::runtime_error("vulkan Concat: missing VulkanContext");

    // ---------- fill params ----------
    ConcatParams p{};
    p.num = static_cast<uint32_t>(tensors.size());
    p.axis = static_cast<uint32_t>(dim);
    p.prefix_sum[0] = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        p.prefix_sum[i + 1] = p.prefix_sum[i] + static_cast<uint32_t>(tensors[i].shape(dim));
    }
    auto out_strides = res.strides();
    for (int i = 0; i < 6; ++i) {
        const size_t slot    = static_cast<size_t>(i);
        p.output_shape[i]    = (slot < out_shape.size()) ? static_cast<uint32_t>(out_shape[i]) : 1;
        p.output_strides[i] =
            (slot < out_strides.size()) ? static_cast<uint32_t>(out_strides[i]) : 0;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto in_strides = tensors[i].strides();
        for (int j = 0; j < 6; ++j) {
            p.input_strides[i][j] = (static_cast<size_t>(j) < in_strides.size())
                                        ? static_cast<uint32_t>(in_strides[j])
                                        : 0;
        }
    }

    // The guard owns the parameter buffer and its device memory; it outlives
    // submitCompute, which waits on its fence before returning.
    auto params_buf = ctx->createBuffer<ConcatParams>(p);

    // ---------- collect input buffers ----------
    std::vector<vk::Buffer> buffers;
    buffers.reserve(10);
    buffers.push_back(params_buf.buffer());
    buffers.push_back(res_impl->buffer());
    for (int i = 0; i < 8; ++i) {
        // Unused slots repeat input 0: the shader never reads them, but every
        // descriptor must be bound.
        const size_t idx = static_cast<size_t>(i) < tensors.size() ? static_cast<size_t>(i) : 0;
        auto impl = std::dynamic_pointer_cast<VKTensor>(tensors[idx].get_impl());
        if (!impl) throw std::runtime_error("vulkan Concat: expected VKTensors");
        buffers.push_back(impl->buffer());
    }

    const uint32_t gx = static_cast<uint32_t>((res.numel() + 255) / 256);
    ctx->submitCompute(OpType::Concat, dtype, buffers, gx, 1, 1, nullptr, 0);
    return res;
}

}  // namespace ops
