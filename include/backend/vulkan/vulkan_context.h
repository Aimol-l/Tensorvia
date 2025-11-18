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

    // matmul_i8/i16/i32/i64/fp16/bfp16/fp32/fp64 --> pipeline
    // relu_i8/i16/i32/i64/fp16/bfp16/fp32/fp64 --> pipeline
    // ...
    std::unordered_map<std::string, vk::Pipeline> m_pipelines; 

    // matmul --> pipeline_layout
    // relu --> pipeline_layout
    // ...
    std::unordered_map<std::string, vk::PipelineLayout> m_pipeline_layouts;


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

public:
    VulkanContext(){
        this->createInstance();
        this->setupDebugMessenger();
        this->choosePhysicalDevice();       // 按优先级选择设备：独显 > 集显 > 其他
        this->createLogicalDevice();
        this->createCommandPool();          // 创建命令池
        this->createDescriptorPool();       // 创建描述符池
        this->createComputePipeline();      // 创建计算管线
        this->createSyncObjects();          // 创建同步对象
        
        // 打印选中的设备信息
        auto props = m_phydevice.getProperties();
        std::cout << "*************************Vulkan Device Info******************" << std::endl;
        std::println("Selected Vulkan Device:{}",props.deviceName);
        std::println("Global memory size: {}MB",props.limits.maxMemoryAllocationCount/(1024 * 1024));
        std::println("Max Compute Work Group Size: [{}]",props.limits.maxComputeWorkGroupSize);
        std::println("Max Compute Work Item Size: [{}]",props.limits.maxComputeWorkGroupCount);
        std::println("Max Compute Work Invocations Size: [{}]",props.limits.maxComputeWorkGroupInvocations);
        std::cout << "***********************************************************" << std::endl;
    }
    ~VulkanContext() override;

    // "matmul","relu",...
    void registerOps(std::string op,int tensor_count,int params_size){
        // 判断算子文件是否存在
        static const std::vector<std::string> REQUIREDTYPE = {
            "i8","i16","i32","i64","fp16","fp32","fp64","bfp16"
        };
        // 创建 pipelineLayout
        this->createPipelineLayout(op,tensor_count,params_size);
        // 加载一个算子的8个不同类型的shader
        for(auto& type:REQUIREDTYPE){
            std::string spvFile = std::format("{}_{}.spv",op,type);
            std::ifstream file(spvFile.c_str());
            if(!file.good()){
                throw std::runtime_error(std::format("{} not found",spvFile));
            }
            // 1. 加载shader
            auto spvCode = readSpvFile(spvFile);
            vk::ShaderModule shaderModule = createShaderModule(spvCode);
            // 2. 配置shader stage
            vk::PipelineShaderStageCreateInfo stageInfo;
            stageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
            stageInfo.setModule(shaderModule);
            stageInfo.setPName("main");
            // 3. 创建计算管线
            vk::ComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = this->m_pipeline_layouts[op];
            auto result = m_device.createComputePipeline(nullptr, pipelineInfo);
            if (result.result != vk::Result::eSuccess) {
                m_device.destroyShaderModule(shaderModule);
                throw std::runtime_error("Failed to create compute pipeline!");
            }
            this->m_pipelines[spvFile] = result.value;
            this->m_device.destroyShaderModule(shaderModule); // ✅ 管线已持有内部副本，可安全销毁
        }

    }
private:
    void createInstance(){
        if (m_enableValidationLayers && !this->checkLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("Tensorvia")
            .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPEngineName("TestEngine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(VK_API_VERSION_1_4);

        std::vector<const char*> instanceExtensions;
        if (m_enableValidationLayers) {
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        vk::InstanceCreateInfo createInfo;
        createInfo.setPApplicationInfo(&appInfo)
            .setPEnabledLayerNames(m_validationLayers)
            .setPEnabledExtensionNames(instanceExtensions);

        this->m_instance = vk::createInstance(createInfo);
    }

    bool checkLayerSupport() {
        auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const char* layerName : m_validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                return false;
            }
        }
        return true;
    }
    bool checkExtensionSupport() {
        auto availableExten = vk::enumerateInstanceExtensionProperties();
        for (const char* extName : m_deviceExtensions) {
            bool found = false;
            for (const auto& prop : availableExten) {
                if (strcmp(extName, prop.extensionName) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::println("Required extension {} not available!", extName);
                return false;
            }
        }
        return true;
    }
    
    void setupDebugMessenger() {
        if (!m_enableValidationLayers) return;
        static PFN_vkDebugUtilsMessengerCallbackEXT DebugUtilsMessengerCallback = [](
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData)->VkBool32 {
                std::println("{}", pCallbackData->pMessage);
                return VK_FALSE;
        };
        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = DebugUtilsMessengerCallback
        };
        pfnDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(static_cast<VkInstance>(m_instance), "vkDestroyDebugUtilsMessengerEXT");
        PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(this->m_instance, "vkCreateDebugUtilsMessengerEXT"));
        if (vkCreateDebugUtilsMessenger) {
            VkResult result = vkCreateDebugUtilsMessenger(this->m_instance, &debugUtilsMessengerCreateInfo, nullptr,&m_debugMessenger);
            if (result)
                std::println("[VulkanBase] ERROR Failed to create a debug messenger! Error code: {}", int32_t(result));
        }
    }

    void choosePhysicalDevice(){
        // 获取所有的支持设备
        auto devices = this->m_instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        auto find_best_compute_family = [](std::vector<vk::QueueFamilyProperties>&& queue_families) -> uint32_t{
            for (uint32_t i = 0; i < queue_families.size(); i++) {
                const auto& family = queue_families[i];
                if (family.queueFlags & vk::QueueFlagBits::eCompute) {
                    if (family.queueFlags & vk::QueueFlagBits::eGraphics) {
                        return i;  // 最优选择，能同时支持图形和计算的队列族性能会更好些。
                    }
                }
            }
            for (uint32_t i = 0; i < queue_families.size(); i++) {
                if (queue_families[i].queueFlags & vk::QueueFlagBits::eCompute) {
                    return i;
                }
            }
            throw std::runtime_error("No compute queue family found!");
        };
        auto find_best_phydevice = [](std::vector<vk::PhysicalDevice>& devices) -> vk::PhysicalDevice {
            std::vector<vk::PhysicalDeviceType> priority_order = {
                vk::PhysicalDeviceType::eDiscreteGpu,vk::PhysicalDeviceType::eIntegratedGpu,
                vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu,vk::PhysicalDeviceType::eOther
            };
            for (auto type : priority_order) {
                for (auto& device : devices) {
                    auto props = device.getProperties();
                    if (props.deviceType != type) continue;
                    bool has_compute = false;
                    auto queue_families = device.getQueueFamilyProperties();
                    for (const auto& family : queue_families) {
                        if (family.queueFlags & vk::QueueFlagBits::eCompute) {
                            has_compute = true;
                            break;
                        }
                    }
                    if (!has_compute)continue;
                    auto features = device.getFeatures();
                    if (!features.shaderInt64)continue;
                    vk::PhysicalDeviceVulkan12Features vulkan12_features;
                    vk::PhysicalDeviceFeatures2 features2;
                    features2.pNext = &vulkan12_features;
                    device.getFeatures2(&features2);
                    auto limits = props.limits;
                    if (limits.maxComputeWorkGroupSize[0] < 16 || limits.maxComputeWorkGroupSize[1] < 16) {
                        continue;
                    }
                    return device;
                }
            }
            throw std::runtime_error("No suitable physical device found that supports compute shaders!");
        };
        auto phydevice = find_best_phydevice(devices);
        auto computefamily = find_best_compute_family(phydevice.getQueueFamilyProperties());
        this->m_phydevice = phydevice;
        this->m_queuefamily = computefamily;
    }

    void createLogicalDevice() {
        if (m_queuefamily == UINT32_MAX) {
            throw std::runtime_error("Queue family index not set! Call choosePhysicalDevice first.");
        }
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo({},m_queuefamily,1, &queuePriority);

        vk::PhysicalDeviceVulkan12Features vulkan12Features = {};
        vulkan12Features.shaderInt8 = VK_TRUE;
        vulkan12Features.shaderFloat16 = VK_TRUE;

        vk::PhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.shaderInt64 = VK_TRUE;

        vk::DeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.setQueueCreateInfos(queueCreateInfo)
                    .setEnabledExtensionCount(static_cast<uint32_t>(m_deviceExtensions.size()))
                    .setPpEnabledExtensionNames(m_deviceExtensions.data())
                    .setPEnabledFeatures(&deviceFeatures);

        deviceCreateInfo.setPNext(&vulkan12Features);

        try {
            this->m_device = m_phydevice.createDevice(deviceCreateInfo);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create logical device: ") + e.what());
        }
        this->m_compute_queue = m_device.getQueue(m_queuefamily, 0);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo;
        poolInfo.setQueueFamilyIndex(m_queuefamily)
                .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient);
        
        try {
            m_command_pool = m_device.createCommandPool(poolInfo);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create command pool: ") + e.what());
        }
    }
    void createDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 3000},      // 存储缓冲区 - 用于所有张量数据
            {vk::DescriptorType::eUniformBuffer, 500}        // 统一缓冲区 - 用于计算参数
        };
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(2000)  // 足够的描述符集数量
                .setPoolSizes(poolSizes)
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        try {
            m_descriptor_pool = m_device.createDescriptorPool(poolInfo);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create descriptor pool: ") + e.what());
        }
    }
    void createSyncObjects() {
        // 预创建一些栅栏
        m_inflight_fences.reserve(10);
    }

    vk::Fence createFence() {
        vk::FenceCreateInfo fenceInfo;
        fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled); // 初始为触发状态
        
        try {
            vk::Fence fence = m_device.createFence(fenceInfo);
            m_inflight_fences.push_back(fence);
            return fence;
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create fence: ") + e.what());
        }
    }

    void cleanupSyncObjects() {
        for (auto& fence : m_inflight_fences) {
            m_device.destroyFence(fence);
        }
        m_inflight_fences.clear();
    }

    // 在 private 中添加
    std::vector<uint32_t> readSpvFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open SPIR-V file: " + filename);
        }
        size_t fileSize = (size_t)file.tellg();
        std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read((char*)buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    vk::ShaderModule createShaderModule(const std::vector<uint32_t>& code) {
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(code);
        return m_device.createShaderModule(createInfo);
    }
    // 为不同 shader 创建不同 pipeline
    vk::Pipeline createComputePipeline(const std::string& spvFile) {
        // 1. 加载 shader
        auto spvCode = readSpvFile(spvFile);
        vk::ShaderModule shaderModule = createShaderModule(spvCode);
        // 2. 配置 shader stage
        vk::PipelineShaderStageCreateInfo stageInfo{};
        stageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";
        // 3. 创建 pipeline
        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = m_pipeline_layout;
        auto result = m_device.createComputePipeline(nullptr, pipelineInfo);
        if (result.result != vk::Result::eSuccess) {
            m_device.destroyShaderModule(shaderModule);
            throw std::runtime_error("Failed to create compute pipeline!");
        }
        vk::Pipeline pipeline = result.value;
        m_device.destroyShaderModule(shaderModule); // ✅ 管线已持有内部副本，可安全销毁
        return pipeline;
    }

    // 初始化时创建一个默认 pipeline（例如 identity）
    void createComputePipeline() {
        // 先创建 pipeline layout
        createPipelineLayout();

        // 示例：加载一个测试 shader
        try {
            m_pipeline[0] = createComputePipeline("shaders/identity.comp.spv");
            std::println("✅ Default compute pipeline created.");
        } catch (const std::exception& e) {
            std::println("⚠️ Failed to create default pipeline: {}", e.what());
            // 可继续运行，后续按需创建
        }
    }
    void createPipelineLayout(std::string op,int tensor_count,int params_size) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        // 创建 descriptor set layout（用于张量输入/输出）
        for(int i = 0;i<tensor_count;i++){
            vk::DescriptorSetLayoutBinding binding0{};
            binding0.binding = i;
            binding0.descriptorType = vk::DescriptorType::eStorageBuffer;
            binding0.descriptorCount = 1;
            binding0.stageFlags = vk::ShaderStageFlagBits::eCompute;
            bindings.push_back(binding0);
        }
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);
        vk::DescriptorSetLayout descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
        // 创建 pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setSetLayouts(descriptorSetLayout);
        // 如果需要 push constants:
        if(params_size != 0){
            vk::PushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
            pushConstantRange.offset = 0;
            pushConstantRange.size = params_size;
            pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);
        }
        this->m_pipeline_layouts[op] = m_device.createPipelineLayout(pipelineLayoutInfo);
    }
};