#include "vulkan_tensor.h"

VKTensor::~VKTensor() {
    if (m_buffer)   m_context->getDevice().destroyBuffer(m_buffer);
    if (m_memory)   m_context->getDevice().freeMemory(m_memory);
}
VKTensor::VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context)
    : m_numel(numel)
    , m_dtype(dtype)
    , m_context(std::move(context))
    , m_buffer(VK_NULL_HANDLE)
    , m_memory(VK_NULL_HANDLE)
{
    vk::Device device = m_context->getDevice();
    vk::PhysicalDevice phyDevice = m_context->getPhysicalDevice();

    size_t size_bytes = numel * calc_dtype_size(dtype);

    // 1. 创建 buffer
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size_bytes)
              .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eTransferDst |   // for clone / copy_from_host
                        vk::BufferUsageFlagBits::eTransferSrc);   // for copy_to / clone
    m_buffer = device.createBuffer(bufferInfo);

    // 2. 分配内存（device-local，最优性能）
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
    if (!found) {
        // fallback: try any host-visible if device-local not available (rare on discrete GPU)
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                memoryTypeIndex = i;
                found = true;
                break;
            }
        }
        if (!found) {
            device.destroyBuffer(m_buffer);
            throw std::runtime_error("Failed to find suitable memory type for VKTensor buffer!");
        }
    }

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size)
             .setMemoryTypeIndex(memoryTypeIndex);
    m_memory = device.allocateMemory(allocInfo);

    // 3. 绑定 buffer 到内存
    device.bindBufferMemory(m_buffer, m_memory, 0);
}

VKTensor::VKTensor(void* ptr, size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context)
    : VKTensor(numel, dtype, context)  // 先分配设备内存
{
    if (!ptr) return;

    size_t size_bytes = numel * calc_dtype_size(dtype);

    // 1. 创建 staging buffer (host-visible)
    vk::Device device = m_context->getDevice();
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMemory;

    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.setSize(size_bytes)
               .setUsage(vk::BufferUsageFlagBits::eTransferSrc);
    stagingBuffer = device.createBuffer(stagingInfo);

    vk::MemoryRequirements stagingReqs = device.getBufferMemoryRequirements(stagingBuffer);
    vk::PhysicalDeviceMemoryProperties memProps = m_context->getPhysicalDevice().getMemoryProperties();

    uint32_t stagingMemoryType = 0;
    bool found = false;
    vk::MemoryPropertyFlags hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((stagingReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & hostFlags) == hostFlags) {
            stagingMemoryType = i;
            found = true;
            break;
        }
    }
    if (!found) {
        device.destroyBuffer(stagingBuffer);
        throw std::runtime_error("Failed to find host-visible memory for staging buffer!");
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
    m_context->computeQueue().submit(submitInfo, fence);  // 需暴露 computeQueue
    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 5. cleanup staging
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), cmd);
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingMemory);
}

void VKTensor::copy_to(TensorImpl &dst) const{
    auto* vk_dst = dynamic_cast<VKTensor*>(&dst);

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
    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 4. Cleanup
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), cmd);
}

std::unique_ptr<TensorImpl> VKTensor::clone() const
{
    return std::unique_ptr<TensorImpl>();
}

std::shared_ptr<ContextImpl> VKTensor::context() const
{
    return std::shared_ptr<ContextImpl>();
}

std::unique_ptr<TensorImpl> VKTensor::clone_as_contiguous(const Metadata &) const
{
    return std::unique_ptr<TensorImpl>();
}
void* VKTensor::data(){
    throw std::runtime_error("VKTensor::data() is not supported. Use copy_to().");
}
const void* VKTensor::data() const {
    throw std::runtime_error("VKTensor::data() is not supported. Use copy_to().");
}