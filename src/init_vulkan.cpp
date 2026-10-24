#include <cstddef>
#include <memory>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/factory.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"

namespace {
// The Vulkan device, pipelines and shader registry are only created when the
// first Device::VULKAN tensor is requested; CPU-only programs never touch the
// driver. This TU exists only in builds with the `vulkan` feature.
struct VulkanLazyRegistrar {
    VulkanLazyRegistrar() {
        register_backend_initializer(Device::VULKAN, [] {
            auto ctx = std::make_shared<VulkanContext>();  // registers pipelines
            register_tensor_impl(
                Device::VULKAN,
                [ctx](void* ptr, size_t numel, DataType dtype) {
                    return std::make_shared<VKTensor>(ptr, numel, dtype, ctx);
                });
        });
    }
} vulkan_lazy_registrar;
}  // namespace
