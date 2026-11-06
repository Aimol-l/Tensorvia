#include <cstddef>
#include <memory>


#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "tensorvia/backend/vulkan/vk_mem_alloc.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace {
struct VulkanLazyRegistrar {
    VulkanLazyRegistrar() {
        register_device_probe(Device::VULKAN, [] {
            return DeviceProbe{VulkanContext::device_count(),
                               VulkanContext::preferred_device_index()};
        });
        register_backend_initializer(Device::VULKAN, [](int index) {
            auto ctx = std::make_shared<VulkanContext>(index);  // 注册流水线
            const DeviceId device{Device::VULKAN, index};
            register_tensor_impl(
                device,
                [ctx](void* ptr, size_t numel, DataType dtype) {
                    return std::make_shared<VKTensor>(ptr, numel, dtype, ctx);
                });
            register_device_control(device, DeviceControl{
                [ctx] { ctx->synchronize(); },
                [ctx] { ctx->synchronize(); },
            });
        });
    }
} vulkan_lazy_registrar;
}  // namespace
