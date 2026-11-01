#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace {
// Vulkan 设备、流水线与着色器注册表在请求第一个 Device::VULKAN 张量时才创建；
// 纯 CPU 程序从不触碰驱动。本 TU 只出现在带 `vulkan` feature 的构建中。
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
            // 提交只入队，synchronize() 才等队列清空（并顺手回收在飞的命令与
            // 退役 buffer）。VMA 随 buffer 释放归还内存，没有额外的空闲块缓存。
            register_device_control(device, DeviceControl{
                [ctx] { ctx->synchronize(); },
                [ctx] { ctx->synchronize(); },
            });
        });
    }
} vulkan_lazy_registrar;
}  // namespace
