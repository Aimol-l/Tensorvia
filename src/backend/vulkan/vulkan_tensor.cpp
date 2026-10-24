#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/context.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"

namespace {

// Owns buffer+memory while init() is still building the object: if init throws
// (no suitable memory type, fence timeout, staging failure) the destructor of
// the object never runs, so raw handles would leak. The success path calls
// release() and hands ownership to the members.
struct VKBufferMemGuard {
    vk::Device       device{};
    vk::Buffer       buffer{};
    vk::DeviceMemory memory{};
    void release() {
        buffer = vk::Buffer{};
        memory = vk::DeviceMemory{};
    }
    ~VKBufferMemGuard() {
        if (memory) device.freeMemory(memory);
        if (buffer) device.destroyBuffer(buffer);
    }
};

}  // namespace

VKTensor::~VKTensor() {
    vk::Device device = m_context->getDevice();
    if (m_buffer) device.destroyBuffer(m_buffer);
    if (m_memory) device.freeMemory(m_memory);
}

void VKTensor::init(void* ptr, size_t numel, DataType dtype,
                    std::shared_ptr<VulkanContext> context) {
    this->m_dtype   = dtype;
    this->m_numel   = numel;
    this->m_context = std::move(context);
    if (!m_context) throw std::runtime_error("VKTensor: null VulkanContext");

    // Shares the device queue/command pool with submitCompute, so it must
    // serialize on the same recursive mutex.
    std::lock_guard<std::recursive_mutex> lock(m_context->submitMutex());
    vk::Device device            = m_context->getDevice();
    vk::PhysicalDevice phyDevice = m_context->getPhysicalDevice();

    // This backend's shaders index with 32-bit integers. This is the single
    // allocation entry point, so the invariant is enforced here rather than in
    // every operator.
    vulkan_detail::require_int32_indexable(static_cast<int64_t>(numel), "VKTensor::init");

    const size_t size_bytes = numel * calc_dtype_size(dtype);

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.setSize(size_bytes)
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                  vk::BufferUsageFlagBits::eTransferDst |   // clone / copy_from_host
                  vk::BufferUsageFlagBits::eTransferSrc);   // copy_to / clone
    VKBufferMemGuard guard{device};
    guard.buffer   = device.createBuffer(bufferInfo);
    this->m_buffer = guard.buffer;

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
    if (!found) throw std::runtime_error("Can't allocate device memory");

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size).setMemoryTypeIndex(memoryTypeIndex);
    guard.memory   = device.allocateMemory(allocInfo);
    this->m_memory = guard.memory;
    device.bindBufferMemory(m_buffer, m_memory, 0);

    if (ptr == nullptr) {
        guard.release();
        return;
    }

    // Stage host data through a host-visible buffer, then copy on the queue.
    void* staging = nullptr;
    {
        auto staging_guard = m_context->createStagingBuffer(size_bytes);
        staging = device.mapMemory(staging_guard.memory(), 0, size_bytes);
        std::memcpy(staging, ptr, size_bytes);
        device.unmapMemory(staging_guard.memory());

        vk::CommandBufferAllocateInfo cmdAlloc{};
        cmdAlloc.setCommandPool(m_context->commandPool())
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
        vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(beginInfo);
        vk::BufferCopy copyRegion{};
        copyRegion.setSize(size_bytes);
        cmd.copyBuffer(staging_guard.buffer(), m_buffer, copyRegion);
        cmd.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(cmd);
        vk::Fence fence = device.createFence({});
        m_context->computeQueue().submit(submitInfo, fence);
        const vk::Result wait_result = vulkan_detail::wait_fence(device, fence);
        device.destroyFence(fence);
        device.freeCommandBuffers(m_context->commandPool(), 1, &cmd);
        vulkan_detail::check_fence(wait_result, "VKTensor::init: waitForFences", __FILE__,
                                   __LINE__);
    }
    guard.release();
}

VKTensor::VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context) {
    this->init(nullptr, numel, dtype, std::move(context));
}

VKTensor::VKTensor(void* ptr, size_t numel, DataType dtype,
                   std::shared_ptr<VulkanContext> context) {
    this->init(ptr, numel, dtype, std::move(context));
}

void VKTensor::copy_to(std::shared_ptr<TensorImpl> dst) const {
    std::lock_guard<std::recursive_mutex> lock(m_context->submitMutex());
    auto vk_dst = std::dynamic_pointer_cast<VKTensor>(dst);
    if (!vk_dst) throw std::invalid_argument("copy_to: destination is not a VKTensor");
    if (m_numel != vk_dst->m_numel || m_dtype != vk_dst->m_dtype) {
        throw std::invalid_argument("Tensor size or dtype mismatch in copy_to!");
    }
    if (m_context != vk_dst->m_context) {
        throw std::invalid_argument("Cannot copy between different Vulkan contexts!");
    }

    const size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    vk::Device device       = m_context->getDevice();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.setCommandPool(m_context->commandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.setSize(size_bytes);
    cmd.copyBuffer(m_buffer, vk_dst->m_buffer, copyRegion);
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_context->computeQueue().submit(submitInfo, fence);
    const vk::Result wait_result = vulkan_detail::wait_fence(device, fence);

    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), 1, &cmd);
    vulkan_detail::check_fence(wait_result, "VKTensor::copy_to: waitForFences", __FILE__,
                               __LINE__);
}

std::unique_ptr<TensorImpl> VKTensor::clone() const {
    // init() below takes the same recursive mutex on this thread.
    std::lock_guard<std::recursive_mutex> lock(m_context->submitMutex());
    auto cloned            = std::make_unique<VKTensor>(m_numel, m_dtype, m_context);
    const size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    vk::Device device       = m_context->getDevice();

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(m_context->commandPool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(beginInfo);
    vk::BufferCopy copyRegion{};
    copyRegion.size = size_bytes;
    cmd.copyBuffer(m_buffer, cloned->buffer(), copyRegion);
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_context->computeQueue().submit(submitInfo, fence);
    // Collect the result before throwing: fence and command buffer must be
    // released either way.
    const vk::Result wait_result = vulkan_detail::wait_fence(device, fence);
    device.destroyFence(fence);
    device.freeCommandBuffers(m_context->commandPool(), 1, &cmd);
    vulkan_detail::check_fence(wait_result, "VKTensor::clone: waitForFences", __FILE__,
                               __LINE__);
    return cloned;
}

std::shared_ptr<ContextImpl> VKTensor::context() const {
    return m_context;
}

std::unique_ptr<TensorImpl> VKTensor::clone_as_contiguous(const Metadata& meta) const {
    auto cloned = std::make_unique<VKTensor>(meta.numel, this->m_dtype, this->m_context);
    vulkan_repack(meta, this->buffer(), cloned->buffer(), this->m_context);
    return cloned;
}

void* VKTensor::data() {
    return m_buffer;
}

const void* VKTensor::data() const {
    return m_buffer;
}

void VKTensor::copy_to_host(void* dst, DataType dtype) const {
    if (!dst) throw std::invalid_argument("VKTensor::copy_to_host: null destination");
    m_context->downloadBuffer(m_buffer, dst, m_numel * calc_dtype_size(dtype));
}

void VKTensor::copy_from_host(const void* src, DataType dtype) {
    if (!src) throw std::invalid_argument("VKTensor::copy_from_host: null source");
    m_context->uploadBuffer(m_buffer, src, m_numel * calc_dtype_size(dtype));
}
