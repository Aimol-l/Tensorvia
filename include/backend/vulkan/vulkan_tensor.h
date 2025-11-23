#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <print>
#include "core/tensor.h"
#include "core/types.h"
#include "core/context.h"
#include "vulkan_constant.h"
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

    void init(void* ptr,size_t numel, DataType dtype,std::shared_ptr<VulkanContext> context);

    void* data();                // 不要具体使用这个指针！！！！ 
    const void* data() const;    // 不要具体使用这个指针！！！！
    vk::Buffer buffer() const { return m_buffer; }      // 用这个！


    size_t numel() const override{return m_numel;}
    void copy_to(std::shared_ptr<TensorImpl> dst) const override;
    std::unique_ptr<TensorImpl> clone() const override;
    std::shared_ptr<ContextImpl> context() const override;
    std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const override;
};