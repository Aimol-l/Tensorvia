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
    VmaAllocation                 m_allocation = nullptr;
    std::shared_ptr<VulkanContext> m_context;

public:
    ~VKTensor() override;
    VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context);
    VKTensor(void* ptr, size_t numel, DataType dtype,
             std::shared_ptr<VulkanContext> context);

    void init(void* ptr, size_t numel, DataType dtype,
              std::shared_ptr<VulkanContext> context);

    // 不要解引用：它们返回 VkBuffer 句柄而非宿主指针。下命令用下面的 buffer()，
    // 宿主访问用 copy_to_host/to_host。
    void*       data() override;
    const void* data() const override;
    vk::Buffer  buffer() const { return m_buffer; }

    size_t numel() const override { return m_numel; }

    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    
    std::unique_ptr<TensorImpl> clone() const;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const;

    // 宿主中转传输（device-local 内存 <-> 宿主）。
    void copy_to_host(void* dst, DataType dtype) const override;
    void copy_from_host(const void* src, DataType dtype) override;
};
