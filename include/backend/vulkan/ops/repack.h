#pragma once
#include "core/types.h"
#include "core/tensor.h"
#include "vulkan_context.h" 

template <Device D> struct RepackImpl;

template <>
struct RepackImpl<Device::VULKAN> {
    static void execute(const Metadata& meta,
        vk::Buffer input,
        vk::Buffer output,
        std::shared_ptr<VulkanContext> ctx
    );
};

// 显式实例化声明
extern template struct RepackImpl<Device::VULKAN>;