#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

VKTensor::~VKTensor() {
    if (m_buffer) m_context->destroyBuffer(m_buffer, m_allocation);
}

void VKTensor::init(void* ptr, size_t numel, DataType dtype,
                    std::shared_ptr<VulkanContext> context) {
    this->m_dtype   = dtype;
    this->m_numel   = numel;
    this->m_context = std::move(context);
    if (!m_context) throw std::runtime_error("VKTensor: null VulkanContext");

    std::lock_guard<std::recursive_mutex> lock(m_context->submitMutex());
    vk::Device device = m_context->getDevice();

    vulkan_detail::require_int32_indexable(static_cast<int64_t>(numel), "VKTensor::init");

    const size_t size_bytes = numel * calc_dtype_size(dtype);

    VKBufferGuard guard = m_context->createDeviceBuffer(
        size_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |   // clone / copy_from_host
            vk::BufferUsageFlagBits::eTransferSrc);   // copy_to / clone
    this->m_buffer     = guard.buffer();
    this->m_allocation = guard.allocation();

    if (ptr == nullptr) {
        guard.release();
        return;
    }

    // Stage host data through a host-visible buffer, then copy on the queue.
    {
        auto staging_guard = m_context->createStagingBuffer(size_bytes);
        std::memcpy(staging_guard.mapped(), ptr, size_bytes);
        vmaFlushAllocation(m_context->allocator(), staging_guard.allocation(), 0, VK_WHOLE_SIZE);

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
        vulkan_detail::check_fence(wait_result, "VKTensor::init: waitForFences", __FILE__,__LINE__);
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
