#pragma once
#include "core/context.h"
#include <vulkan/vulkan.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <print>

class VulkanContext : public ContextImpl {

private:
    vk::Device m_device;
    vk::Instance m_instance;
    vk::PhysicalDevice m_phydevice;

    uint32_t m_queuefamily;
    vk::Queue m_compute_queue;

    vk::CommandPool m_command_pool;
    vk::DescriptorPool m_descriptor_pool;

    vk::PipelineCache m_pipeline_cache_handle;
    std::unordered_map<size_t, vk::Pipeline> m_pipeline_cache;

    // 添加同步对象
    std::vector<vk::Fence> m_inflight_fences;
    std::mutex m_pipeline_cache_mutex;


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
        this->choosePhysicalDevice(); // 按优先级选择设备：独显 > 集显 > 其他
        this->createLogicalDevice();
        this->createCommandPool();  // 创建命令池
        this->createDescriptorPool();   // 创建描述符池
        this->createPipelineCache();    // 创建计算管线
        this->createSyncObjects();      // 创建同步对象
        
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

private:
    void createInstance(){
        if (m_enableValidationLayers && ! this->checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("Tensorvia")
            .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
            .setPEngineName("TestEngine")
            .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
            .setApiVersion(VK_API_VERSION_1_4);

        vk::InstanceCreateInfo createInfo;
        createInfo.setPApplicationInfo(&appInfo).setPEnabledLayerNames(m_validationLayers); // 纯通用计算就只要开启验证层就行了
        this->m_instance =  vk::createInstance(createInfo);
    }

    bool checkValidationLayerSupport() {
        auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const char* layerName : m_validationLayers) {
            std::string name(layerName);
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
                    if (limits.maxComputeWorkGroupSize[0] < 16 || 
                        limits.maxComputeWorkGroupSize[1] < 16) {
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
        // 描述符池大小配置 - 根据你的需求调整
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            {vk::DescriptorType::eStorageBuffer, 1000},      // 存储缓冲区
            {vk::DescriptorType::eUniformBuffer, 100},       // 统一缓冲区  
            {vk::DescriptorType::eCombinedImageSampler, 100}, // 图像采样器
            {vk::DescriptorType::eStorageImage, 100}         // 存储图像
        };

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(1000)  // 最大描述符集数量
                .setPoolSizes(poolSizes)
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

        try {
            m_descriptor_pool = m_device.createDescriptorPool(poolInfo);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create descriptor pool: ") + e.what());
        }
    }

    void createPipelineCache() {
        vk::PipelineCacheCreateInfo cacheInfo;
        try {
            m_pipeline_cache_handle = m_device.createPipelineCache(cacheInfo);
        } catch (const vk::SystemError& e) {
            throw std::runtime_error(std::string("Failed to create pipeline cache: ") + e.what());
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
};