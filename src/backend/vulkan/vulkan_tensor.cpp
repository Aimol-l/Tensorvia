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
    if (m_buffer) m_context->destroyBuffer(m_buffer, m_allocation, m_queue);
}

void VKTensor::init(void* ptr, size_t numel, DataType dtype,
                    std::shared_ptr<VulkanContext> context) {
    this->m_dtype   = dtype;
    this->m_numel   = numel;
    this->m_context = std::move(context);
    if (!m_context) throw std::runtime_error("VKTensor: null VulkanContext");

    // 先领取本线程的通道：buffer 归属它，之后的提交与释放都要走同一条。
    m_queue = &m_context->currentQueue();
    std::lock_guard<std::recursive_mutex> lock(m_queue->mutex);
    vk::Device device = m_context->getDevice();

    vulkan_detail::require_int32_indexable(static_cast<int64_t>(numel), "VKTensor::init");

    const size_t size_bytes = numel * calc_dtype_size(dtype);
    // VMA 拒绝零尺寸 buffer；空张量保留 1 字节占位，使 data() 保持非空且每个
    // buffer 都可绑定。
    const size_t alloc_bytes = size_bytes == 0 ? 1 : size_bytes;

    VKBufferGuard guard = m_context->createDeviceBuffer(
        alloc_bytes,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |   // 写目标：clone / copy_from_host
            vk::BufferUsageFlagBits::eTransferSrc);   // 读源：copy_to / clone
    this->m_buffer     = guard.buffer();
    this->m_allocation = guard.allocation();

    if (ptr == nullptr || size_bytes == 0) {
        guard.release();
        return;
    }

    // 宿主数据先经宿主可见 buffer 暂存，再在队列上拷贝。
    {
        auto staging_guard = m_context->createStagingBuffer(size_bytes);
        std::memcpy(staging_guard.mapped(), ptr, size_bytes);
        vmaFlushAllocation(m_context->allocator(), staging_guard.allocation(), 0, VK_WHOLE_SIZE);

        vk::CommandBufferAllocateInfo cmdAlloc{};
        cmdAlloc.setCommandPool(*m_context->currentQueue().command_pool)
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

        // 只入队；暂存 buffer 的销毁由 context 排到这次拷贝完成之后。
        m_context->submitAsync(cmd);
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
    std::lock_guard<std::recursive_mutex> lock(m_context->currentQueue().mutex);
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
    allocInfo.setCommandPool(*m_context->currentQueue().command_pool)
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

    m_context->submitAsync(cmd);
}

std::unique_ptr<TensorImpl> VKTensor::clone() const {
    // 下面的 init() 会在此线程上取同一把递归锁。
    std::lock_guard<std::recursive_mutex> lock(m_context->currentQueue().mutex);
    auto cloned            = std::make_unique<VKTensor>(m_numel, m_dtype, m_context);
    const size_t size_bytes = m_numel * calc_dtype_size(m_dtype);
    vk::Device device       = m_context->getDevice();

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(*m_context->currentQueue().command_pool)
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

    m_context->submitAsync(cmd);
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
