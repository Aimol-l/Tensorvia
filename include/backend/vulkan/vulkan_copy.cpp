
#include "core/factory.h"
#include "vulkan_tensor.h"
#include "cpu_tensor.h"
// vulkan --> cpu
void copy_device_to_host(std::shared_ptr<TensorImpl> src, std::shared_ptr<TensorImpl> dst,DataType dtype) {
    // 确保 src 是 VKTensor，dst 是 CPUTensor
    auto* vk_src = dynamic_cast<VKTensor*>(src.get());
    auto* cpu_dst = dynamic_cast<CPUTensor*>(dst.get());
    if (!vk_src || !cpu_dst) {
        throw std::invalid_argument("copy_device_to_host: src must be VKTensor, dst must be CPUTensor");
    }

    size_t size_bytes = src->numel() * calc_dtype_size(dtype);
    auto context = std::dynamic_pointer_cast<VulkanContext>(vk_src->context());
    if (!context) {
        throw std::runtime_error("Invalid VulkanContext");
    }

    vk::Device device = context->getDevice();

    // 1. 创建 staging buffer (host-visible)
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.setSize(size_bytes)
               .setUsage(vk::BufferUsageFlagBits::eTransferDst)
               .setSharingMode(vk::SharingMode::eExclusive);
    vk::Buffer stagingBuffer = device.createBuffer(stagingInfo);

    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(stagingBuffer);
    vk::PhysicalDeviceMemoryProperties memProps = context->getPhysicalDevice().getMemoryProperties();

    uint32_t memoryType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            memoryType = i;
            break;
        }
    }
    if (memoryType == UINT32_MAX) {
        device.destroyBuffer(stagingBuffer);
        throw std::runtime_error("Failed to find host-visible memory for staging buffer");
    }

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size)
             .setMemoryTypeIndex(memoryType);
    vk::DeviceMemory stagingMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    // 2. Record copy: device buffer -> staging buffer
    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(context->commandPool())
           .setLevel(vk::CommandBufferLevel::ePrimary)
           .setCommandBufferCount(1);
    vk::CommandBuffer cmdBuf = device.allocateCommandBuffers(cmdAlloc)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuf.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size_bytes);
    cmdBuf.copyBuffer(vk_src->buffer(), stagingBuffer, copyRegion);
    cmdBuf.end();

    // 3. Submit and wait
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmdBuf);
    vk::FenceCreateInfo fenceInfo{};
    vk::Fence fence = device.createFence(fenceInfo);
    context->computeQueue().submit(submitInfo, fence);
    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 4. Map staging buffer and copy to CPU tensor
    void* mapped = device.mapMemory(stagingMemory, 0, size_bytes);
    std::memcpy(cpu_dst->data(), mapped, size_bytes);
    device.unmapMemory(stagingMemory);

    // 5. Cleanup
    device.destroyFence(fence);
    device.freeCommandBuffers(context->commandPool(), cmdBuf);
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingMemory);
}
// cpu --> vulkan
void copy_host_to_device(std::shared_ptr<TensorImpl> src, std::shared_ptr<TensorImpl> dst,DataType dtype) {

    auto* cpu_src = dynamic_cast<CPUTensor*>(src.get());
    auto* vk_dst = dynamic_cast<VKTensor*>(dst.get());
    if (!cpu_src || !vk_dst) {
        throw std::invalid_argument("copy_host_to_device: src must be CPUTensor, dst must be VKTensor");
    }

    size_t size_bytes = src->numel() * calc_dtype_size(dtype);
    auto context = std::dynamic_pointer_cast<VulkanContext>(vk_dst->context());
    if (!context) {
        throw std::runtime_error("Invalid VulkanContext");
    }

    vk::Device device = context->getDevice();

    // 1. Create staging buffer
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.setSize(size_bytes)
               .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
               .setSharingMode(vk::SharingMode::eExclusive);
    vk::Buffer stagingBuffer = device.createBuffer(stagingInfo);

    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(stagingBuffer);
    vk::PhysicalDeviceMemoryProperties memProps = context->getPhysicalDevice().getMemoryProperties();

    uint32_t memoryType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
            (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            memoryType = i;
            break;
        }
    }
    if (memoryType == UINT32_MAX) {
        device.destroyBuffer(stagingBuffer);
        throw std::runtime_error("Failed to find host-visible memory for staging buffer");
    }

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size)
             .setMemoryTypeIndex(memoryType);
    vk::DeviceMemory stagingMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    // 2. Copy CPU data to staging buffer
    void* mapped = device.mapMemory(stagingMemory, 0, size_bytes);
    std::memcpy(mapped, cpu_src->data(), size_bytes);
    device.unmapMemory(stagingMemory);

    // 3. Record copy: staging -> device buffer
    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(context->commandPool())
           .setLevel(vk::CommandBufferLevel::ePrimary)
           .setCommandBufferCount(1);
    vk::CommandBuffer cmdBuf = device.allocateCommandBuffers(cmdAlloc)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuf.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size_bytes);
    cmdBuf.copyBuffer(stagingBuffer, vk_dst->buffer(), copyRegion);
    cmdBuf.end();

    // 4. Submit and wait
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmdBuf);
    vk::Fence fence = device.createFence({});
    context->computeQueue().submit(submitInfo, fence);
    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 5. Cleanup
    device.destroyFence(fence);
    device.freeCommandBuffers(context->commandPool(), cmdBuf);
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingMemory);
}