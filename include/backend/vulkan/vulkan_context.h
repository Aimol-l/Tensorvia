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

    mutable std::mutex m_submit_mutex;

    // relu_int32   -->  pipeline 
    //❌不复用,shader 不同（因为 dtype 不同）→ 必须重新创建
    std::unordered_map<std::string, vk::Pipeline> m_pipelines;  

    // relu_int32  -->  pipeline_layout
    // ⚠️可能可复用,如果 push constant 不变，可以复用
    std::unordered_map<std::string, vk::PipelineLayout> m_pipeline_layouts; 

    // relu   -->  descriptor_set_layout
    // ✔️可复用,只是资源结构，不包含 dtype 信息
    std::unordered_map<std::string, vk::DescriptorSetLayout> m_descriptor_set_layouts; 

    bool m_enableValidationLayers = true;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;
    const std::vector<const char*> m_validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*> m_deviceExtensions = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_8bit_storage",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_subgroup_extended_types"
    };

public:
    VulkanContext();
    ~VulkanContext() {};
    void* ctx_raw_ptr() override{
        throw std::runtime_error("no ctx_raw_ptr!");
    }

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
    void registerOp(OpType ops,std::vector<DataType>& Dtypes,int tensor_count, int params_size);
    void registerOp(OpType ops,DataType Dtype,int tensor_count, int params_size);


    template<typename T>
    vk::Buffer createBuffer(const T data){
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.setSize(sizeof(T))
                  .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                            vk::BufferUsageFlagBits::eTransferDst |   // for clone / copy_from_host
                            vk::BufferUsageFlagBits::eTransferSrc);   // for copy_to / clone
        // 1. 创建 buffer
        vk::Buffer buffer = m_device.createBuffer(bufferInfo);
        // 2. 分配显存
        vk::MemoryRequirements memReqs = m_device.getBufferMemoryRequirements(buffer);
        vk::PhysicalDeviceMemoryProperties memProps = m_phydevice.getMemoryProperties();
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
            m_device.destroyBuffer(buffer);
            throw std::runtime_error("Can't allocate device memory");
        }
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.setAllocationSize(memReqs.size).setMemoryTypeIndex(memoryTypeIndex);
        vk::DeviceMemory memory = m_device.allocateMemory(allocInfo);
        m_device.bindBufferMemory(buffer, memory, 0);
        // 3. 创建暂存区
        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingMemory;
        vk::BufferCreateInfo stagingInfo{};
        stagingInfo.setSize(sizeof(T)).setUsage(vk::BufferUsageFlagBits::eTransferSrc);
        stagingBuffer = m_device.createBuffer(stagingInfo);
        vk::MemoryRequirements stagingReqs = m_device.getBufferMemoryRequirements(stagingBuffer);
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
        // 4. copy host data to staging buffer
        void* mappedData = device.mapMemory(stagingMemory, 0, sizeof(T));
        std::memcpy(mappedData, &data, sizeof(T));
        device.unmapMemory(stagingMemory);
        // 5. copy staging buffer to device buffer
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.setCommandPool(m_command_pool)
                 .setLevel(vk::CommandBufferLevel::ePrimary)
                 .setCommandBufferCount(1);
        vk::CommandBuffer cmd = device.allocateCommandBuffers(allocInfo)[0];
        vk::CommandBufferBeginInfo beginInfo{};
        beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd.begin(beginInfo);
        vk::BufferCopy copyRegion{};
        copyRegion.setSize(sizeof(T));
        cmd.copyBuffer(stagingBuffer, buffer, copyRegion);
        cmd.end();
        // 6. submit and wait
        vk::SubmitInfo submitInfo{};
        submitInfo.setCommandBuffers(cmd);
        vk::Fence fence = m_device.createFence({});
        m_compute_queue.submit(submitInfo, fence);
        vk::Result res = m_device.waitForFences(fence, VK_TRUE, UINT64_MAX);
        // 7. cleanup staging
        m_device.destroyFence(fence);
        m_device.freeCommandBuffers(m_command_pool, cmd);
        device.freeMemory(stagingMemory);
        device.destroyBuffer(stagingBuffer);
        // 8. return buffer and memory
        return buffer;
    }

    // 高层接口：用户只需传 buffer
    void submitCompute(
        OpType op,
        DataType dtype,
        const std::vector<vk::Buffer>& buffers,
        uint32_t gx, uint32_t gy, uint32_t gz,
        const void* push_constants,
        size_t push_size
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

    void createDescriptorSetLayout(std::string op,int tensor_count,int params_size);
    void createPipelineLayout(std::string type_op,std::string ori_op,int tensor_count,int params_size);

};