#pragma once
#include "core/context.h"
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <print>
#include <fstream>

class VulkanContext : public ContextImpl {

private:
    vk::Device m_device;
    vk::Instance m_instance;
    vk::PhysicalDevice m_phydevice;

    uint32_t m_queuefamily;
    vk::Queue m_compute_queue;

    vk::CommandPool m_command_pool;
    vk::DescriptorPool m_descriptor_pool;
    std::vector<vk::Fence> m_inflight_fences;

    // relu_i8/i16/i32/i64/fp16/bfp16/fp32/fp64 --> pipeline
    // matmul_i8/i16/i32/i64/fp16/bfp16/fp32/fp64 --> pipeline
    std::unordered_map<std::string, vk::Pipeline> m_pipelines; 

    // relu --> pipeline_layout
    // matmul --> pipeline_layout
    std::unordered_map<std::string, vk::PipelineLayout> m_pipeline_layouts;
    std::unordered_map<std::string, vk::DescriptorSetLayout> m_descriptor_set_layouts;

    bool m_enableValidationLayers = true;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;
    const std::vector<const char*> m_validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> m_deviceExtensions = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_shader_int64",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_8bit_storage",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_subgroup_extended_types"
    };
    std::string makePipelineKey(OpType op, DataType dt) {
        // eg: "matmul_int32"
        return std::format("{}_{}", op_to_string(op), dtype_to_string(dt));
    }
public:
    VulkanContext();
    ~VulkanContext() {};

    vk::Device getDevice()const{
        return m_device;
    }
    vk::PhysicalDevice getPhysicalDevice() const { 
        return m_phydevice; 
    }
    vk::CommandPool commandPool() const { return m_command_pool; }
    vk::Queue computeQueue() const { return m_compute_queue; }
    // 注册算子和背后对应的计算管线。
    // 创建管线需要使用管线布局，管线布局负责描述使用什么类型的shader(vert/frag/compute),确定使用多少输入数据(binding)
    void registerOp(OpType ops, int tensor_count, int params_size);

    // 高层接口：用户只需传 buffer
    void submitCompute(
        const std::string& op_name,
        const std::vector<vk::Buffer>& buffers,
        uint32_t gx, uint32_t gy, uint32_t gz,
        const void* push_constants = nullptr,
        size_t push_size = 0
    );

private:
    void createInstance();
    void setupDebugMessenger();
    void choosePhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createDescriptorPool();

    bool checkLayerSupport();
    bool checkExtensionSupport();

    std::vector<uint32_t> readSpvFile(const std::string& filename);
    void createPipelineLayout(std::string op,int tensor_count,int params_size);

};