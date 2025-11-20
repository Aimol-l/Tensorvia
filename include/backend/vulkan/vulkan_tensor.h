#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include "core/tensor.h"
#include "core/types.h"
#include "core/context.h"
#include "vulkan_context.h" 

class VKTensor : public TensorImpl {
private:
    size_t m_numel;
    DataType m_dtype;
    vk::Buffer m_buffer; 
    vk::DeviceMemory m_memory;
    std::shared_ptr<VulkanContext> m_context;
public:
    ~VKTensor();
    VKTensor(size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context);
    VKTensor(void* ptr, size_t numel, DataType dtype, std::shared_ptr<VulkanContext> context);
    void* data();
    vk::Buffer buffer() const { return m_buffer; }
    size_t numel() const override{return m_numel;}
    const void* data() const;
    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;
};