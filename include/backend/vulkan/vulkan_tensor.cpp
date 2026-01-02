#include "vulkan_tensor.h"
#include "ops/repack.h"
using namespace via;

VKTensor::~VKTensor() {
    if (m_buffer)   m_context->getDevice().destroyBuffer(m_buffer);
    if (m_memory)   m_context->getDevice().freeMemory(m_memory);
}

void VKTensor::init(void *ptr, size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context){
    // 创建gpu buffer
    this->m_dtype = dtype;
    this->m_numel = numel;
    this->m_context = context;
    vk::Device device = m_context->getDevice();
    vk::PhysicalDevice phyDevice = m_context->getPhysicalDevice();
    size_t size_bytes = numel * calc_dtype_size(dtype);
    // 1. 创建 buffer
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size_bytes)
              .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eTransferDst |   // for clone / copy_from_host
                        vk::BufferUsageFlagBits::eTransferSrc);   // for copy_to / clone
    this->m_buffer = device.createBuffer(bufferInfo);
    // 2. 分配显存
    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(m_buffer);
    vk::PhysicalDeviceMemoryProperties memProps = phyDevice.getMemoryProperties();
    uint32_t memoryTypeIndex = 0;
    bool found = false;
    vk::MemoryPropertyFlags desiredFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & desiredFlags) == desiredFlags) {
            memoryTypeIndex = i;
            found = true;
            break;
        }
    }
    if(!found){
        device.destroyBuffer(m_buffer);
        throw std::runtime_error("Can't allocate device memory");
    }
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size).setMemoryTypeIndex(memoryTypeIndex);
    this->m_memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(m_buffer, m_memory, 0);

    if(ptr == nullptr){
        return;
    }
    // 创建暂存区
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMemory;
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.setSize(size_bytes).setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    stagingBuffer = device.createBuffer(stagingInfo);
    vk::MemoryRequirements stagingReqs = device.getBufferMemoryRequirements(stagingBuffer);
    uint32_t stagingMemoryType = 0;
    found = false;
    vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((stagingReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & hostFlags) == hostFlags) {
            stagingMemoryType = i;
            found = true;
            break;
        }
    }
    if(!found){
        device.destroyBuffer(stagingBuffer);
        throw std::runtime_error("Can't allocate host memory");
    }
    vk::MemoryAllocateInfo stagingAlloc{};
    stagingAlloc.setAllocationSize(stagingReqs.size)
                .setMemoryTypeIndex(stagingMemoryType);
    stagingMemory = device.allocateMemory(stagingAlloc);
    device.bindBufferMemory(stagingBuffer, stagingMemory, 0);
    // 2. copy host data to staging buffer
    void* mapped = device.mapMemory(stagingMemory, 0, size_bytes);
    std::memcpy(mapped, ptr, size_bytes);
    device.unmapMemory(stagingMemory);
    // 3. copy staging buffer to device buffer
    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(m_context->commandPool())  // 需要暴露 commandPool
           .setLevel(vk::CommandBufferLevel::ePrimary)
           .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size_bytes);
    cmd.copyBuffer(stagingBuffer, m_buffer, copyRegion);
    cmd.end();

    // 4. submit and wait
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_context->computeQueue().submit(submitInfo, fence);
    vk::Result res = device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    // 5. cleanup staging
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), cmd);
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingMemory);
}
VKTensor::VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context){
    this->init(nullptr,numel,dtype,context);
}

VKTensor::VKTensor(void* ptr, size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context){
    this->init(ptr,numel,dtype,context);
}

void VKTensor::copy_to(std::shared_ptr<TensorImpl> dst) const{
    // auto* vk_dst = dynamic_cast<VKTensor*>(&dst);
    auto vk_dst = std::dynamic_pointer_cast<VKTensor>(dst);
    if (m_numel != vk_dst->m_numel || m_dtype != vk_dst->m_dtype) {
        throw std::invalid_argument("Tensor size or dtype mismatch in copy_to!");
    }
    if (m_context != vk_dst->m_context) {
        throw std::invalid_argument("Cannot copy between different Vulkan contexts!");
    }

    size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    vk::Device device = m_context->getDevice();

    // 1. Allocate command buffer
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.setCommandPool(m_context->commandPool())
             .setLevel(vk::CommandBufferLevel::ePrimary)
             .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(allocInfo)[0];

    // 2. Record copy command
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size_bytes);
    cmd.copyBuffer(m_buffer, vk_dst->m_buffer, copyRegion);
    cmd.end();

    // 3. Submit and wait
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_context->computeQueue().submit(submitInfo, fence);
    vk::Result res = device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 4. Cleanup
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), cmd);
}

std::unique_ptr<TensorImpl> VKTensor::clone() const {
    auto cloned = std::make_unique<VKTensor>(m_numel, m_dtype, m_context);
    size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    auto device = m_context->getDevice();
    // 1. allocate command buffer
    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(m_context->commandPool())
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];
    // 2. record copy
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.size = size_bytes;
    cmd.copyBuffer(m_buffer, cloned->buffer(), copyRegion);
    cmd.end();
    // 3. submit
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_context->computeQueue().submit(submitInfo, fence);
    vk::Result res = device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("VKTensor::clone fence wait failed");
    }
    // 4. cleanup
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), 1, &cmd);
    return cloned;
}

std::shared_ptr<ContextImpl> VKTensor::context() const{
    return m_context;
}

std::unique_ptr<TensorImpl> VKTensor::clone_as_contiguous(const Metadata &meta) const{
    auto cloned = std::make_unique<VKTensor>(meta.numel, this->m_dtype, this->m_context);
    RepackImpl<Device::VULKAN>::execute(meta,this->buffer(),cloned->buffer(),this->m_context);
    return cloned;
}
void* VKTensor::data(){
    return m_buffer;
}
const void* VKTensor::data() const {
    return m_buffer;

}