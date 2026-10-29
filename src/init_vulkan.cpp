#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace {
// The Vulkan device, pipelines and shader registry are only created when the
// first Device::VULKAN tensor is requested; CPU-only programs never touch the
// driver. This TU exists only in builds with the `vulkan` feature.
struct VulkanLazyRegistrar {
    VulkanLazyRegistrar() {
        register_device_probe(Device::VULKAN, [] {
            return DeviceProbe{VulkanContext::device_count(),
                               VulkanContext::preferred_device_index()};
        });
        register_backend_initializer(Device::VULKAN, [](int index) {
            auto ctx = std::make_shared<VulkanContext>(index);  // registers pipelines
            const DeviceId device{Device::VULKAN, index};
            register_tensor_impl(
                device,
                [ctx](void* ptr, size_t numel, DataType dtype) {
                    return std::make_shared<VKTensor>(ptr, numel, dtype, ctx);
                });
            // submitCompute waits on a fence before it returns, so there is
            // never work outstanding to wait for; VMA releases a buffer's
            // memory when the buffer is destroyed, so there is no cache of
            // free blocks to return either.
            register_device_control(device, DeviceControl{[] {}, [] {}});
        });
    }
} vulkan_lazy_registrar;
}  // namespace
