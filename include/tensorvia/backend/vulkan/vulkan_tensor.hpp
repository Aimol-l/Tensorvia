#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.hpp>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"

class VKTensor : public TensorImpl {
private:
    size_t                        m_numel;
    DataType                      m_dtype;
    vk::Buffer                    m_buffer;
    vk::DeviceMemory              m_memory;
    std::shared_ptr<VulkanContext> m_context;

public:
    ~VKTensor() override;
    VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context);
    VKTensor(void* ptr, size_t numel, DataType dtype,
             std::shared_ptr<VulkanContext> context);

    void init(void* ptr, size_t numel, DataType dtype,
              std::shared_ptr<VulkanContext> context);

    // Do NOT dereference these: they return the VkBuffer handle, not a host
    // pointer. Use buffer() below for commands, or copy_to_host/to_host for
    // host access.
    void*       data() override;
    const void* data() const override;
    vk::Buffer  buffer() const { return m_buffer; }

    size_t numel() const override { return m_numel; }

    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;

    // Host staging transfers (device-local memory -> host and back).
    void copy_to_host(void* dst, DataType dtype) const override;
    void copy_from_host(const void* src, DataType dtype) override;
};
