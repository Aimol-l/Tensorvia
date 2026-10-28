#include <spirv/spv_registry.h>
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"

using vulkan_detail::check_fence;
using vulkan_detail::check_result;
using vulkan_detail::wait_fence;

VulkanContext::VulkanContext() {
    this->createInstance();
    this->setupDebugMessenger();
    this->choosePhysicalDevice();
    this->createLogicalDevice();
    this->createCommandPool();
    this->createDescriptorPool();
    this->registerPipelines();

    // 设备信息只属于调试输出：release（NDEBUG）不能污染调用方的 stdout。
    // 与 CPUContext 的同一段打印保持一致。
#ifndef NDEBUG
    auto props    = m_phydevice.getProperties();
    auto memProps = m_phydevice.getMemoryProperties();
    std::println("*************************Vulkan Device Info******************");
    std::println("Selected Vulkan Device: {}", std::string_view(props.deviceName));
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        auto heap = memProps.memoryHeaps[i];
        bool isDeviceLocal =
            (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
        std::println("Memory Heap[{}]: {} MB ({})", i, heap.size / (1024 * 1024),
                     isDeviceLocal ? "DeviceLocal (VRAM)" : "HostVisible (System RAM)");
    }
    std::println("Max Compute Work Group Size: {}", props.limits.maxComputeWorkGroupSize);
    std::println("Max Compute Work Item Size: {}", props.limits.maxComputeWorkGroupCount);
    std::println("Max Compute Work Invocations Size: {}",
                 props.limits.maxComputeWorkGroupInvocations);
    std::println("*************************************************************");
#endif
}

VulkanContext::~VulkanContext() {
    // All resource destruction is handled by the Unique* members and
    // DebugMessenger (declaration order is documented in the partition).
    // Destroying the device while submissions are in flight is UB; every
    // submission waits on its fence, so this is just the final guard.
    if (m_device) {
        try {
            m_device->waitIdle();
        } catch (const vk::SystemError&) {
        }
    }
}

void VulkanContext::registerOp(OpType ops, DataType dtype, int tensor_count, int params_size) {
    std::vector<DataType> types = {dtype};
    this->registerOp(ops, types, tensor_count, params_size);
}

void VulkanContext::registerOp(OpType ops, std::vector<DataType>& dtypes,
                               int tensor_count, int params_size) {
    std::string ori_op = op_to_string(ops);
    for (auto& type : dtypes) {
        std::string key = std::format("{}_{}", ori_op, dtype_to_string(type));
        auto spvCode = vkspv::get_spv_code(key);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(spvCode);

        vk::UniqueShaderModule shaderModule;
        try {
            shaderModule =
                vk::UniqueShaderModule(m_device->createShaderModule(createInfo), *m_device);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            throw;
        }

        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        stageInfo.setModule(*shaderModule);
        stageInfo.setPName("main");

        std::string type_op = std::format("{}_{}", ori_op, dtype_to_string(type));
        this->createDescriptorSetLayout(ori_op, tensor_count, params_size);
        this->createPipelineLayout(type_op, ori_op, tensor_count, params_size);

        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage  = stageInfo;
        pipelineInfo.layout = *this->m_pipeline_layouts[type_op];
        auto result = this->m_device->createComputePipeline(nullptr, pipelineInfo);
        if (result.result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to create compute pipeline!");
        }
        this->m_pipelines[type_op] = vk::UniquePipeline(result.value, *m_device);
    }
}

void VulkanContext::registerPipelines() {
    static std::vector<DataType> AllTypes = {
        DataType::INT8,    DataType::INT16,   DataType::INT32,   DataType::INT64,
        DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT32, DataType::FLOAT64};
    static std::vector<DataType> FloatTypes = {
        DataType::FLOAT16, DataType::BFLOAT16, DataType::FLOAT32, DataType::FLOAT64};

    registerOp(OpType::Add, AllTypes, 1, sizeof(ValueParams<float>));
    registerOp(OpType::AddVec, AllTypes, 3, sizeof(BcastVecParams));

    registerOp(OpType::Sub, AllTypes, 1, sizeof(ValueParams<float>));
    registerOp(OpType::SubVec, AllTypes, 3, sizeof(BcastVecParams));

    registerOp(OpType::Mul, AllTypes, 1, sizeof(ValueParams<float>));
    registerOp(OpType::MulVec, AllTypes, 3, sizeof(BcastVecParams));

    registerOp(OpType::Div, AllTypes, 1, sizeof(ValueParams<float>));
    registerOp(OpType::DivVec, AllTypes, 3, sizeof(BcastVecParams));

    registerOp(OpType::Sin, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Cos, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Tan, FloatTypes, 2, sizeof(int64_t));

    registerOp(OpType::Abs, AllTypes, 2, sizeof(int64_t));
    registerOp(OpType::Exp, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Sqrt, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Pow, FloatTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Log, FloatTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Clamp, AllTypes, 2, sizeof(ClampParams));

    registerOp(OpType::Relu, AllTypes, 1, sizeof(int64_t));
    registerOp(OpType::Silu, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Tanh, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Sigmoid, FloatTypes, 2, sizeof(int64_t));
    // Softmax shaders read integer inputs and write floating point, so the
    // pipeline key must be the *input* dtype; registering only FloatTypes
    // would leave integer inputs without a pipeline.
    registerOp(OpType::Softmax, AllTypes, 2, sizeof(SoftmaxParams));

    registerOp(OpType::Sum, AllTypes, 2, sizeof(int64_t));
    registerOp(OpType::SumVec, AllTypes, 2, sizeof(SoftmaxParams));

    registerOp(OpType::Mean, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::Min, AllTypes, 2, sizeof(int64_t));
    registerOp(OpType::MinVec, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::Argmax, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::Argmin, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::Max, AllTypes, 2, sizeof(int64_t));
    registerOp(OpType::MaxVec, AllTypes, 2, sizeof(SoftmaxParams));

    registerOp(OpType::All, AllTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Any, AllTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Matmul, AllTypes, 3, sizeof(MatmulParams));

    registerOp(OpType::Equal, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::NotEqual, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::Greater, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::GreaterEqual, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::Less, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::LessEqual, AllTypes, 3, sizeof(BcastVecParams));
    registerOp(OpType::Nonzero, AllTypes, 2, sizeof(int64_t));

    registerOp(OpType::Random, AllTypes, 1, sizeof(RandomParams));
    registerOp(OpType::Fill, AllTypes, 1, sizeof(ValueParams<float>));
    registerOp(OpType::Concat, AllTypes, 10, 0);
    registerOp(OpType::Slice, AllTypes, 2, sizeof(SliceParams));
    registerOp(OpType::Repack, DataType::INT8, 2, sizeof(RepackParams));
    registerOp(OpType::Transpose2d, AllTypes, 2, sizeof(Trans2DParams));
    registerOp(OpType::TransposeNd, AllTypes, 2, sizeof(TransNDParams));
}

void VulkanContext::createDescriptorSetLayout(std::string ori_op, int tensor_count,
                                              [[maybe_unused]] int params_size) {
    // Descriptor set layouts carry no dtype information, so they are keyed by
    // the operator name only and shared by every dtype variant.
    if (m_descriptor_set_layouts.contains(ori_op)) return;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(tensor_count);
    for (int i = 0; i < tensor_count; ++i) {
        vk::DescriptorSetLayoutBinding b{};
        b.binding         = i;
        b.descriptorType  = vk::DescriptorType::eStorageBuffer;
        b.descriptorCount = 1;
        b.stageFlags      = vk::ShaderStageFlagBits::eCompute;
        bindings.push_back(b);
    }
    vk::UniqueDescriptorSetLayout descriptorSetLayout;
    try {
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);
        descriptorSetLayout = vk::UniqueDescriptorSetLayout(
            m_device->createDescriptorSetLayout(layoutInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("createDescriptorSetLayout failed: ") + e.what());
    }
    this->m_descriptor_set_layouts[ori_op] = std::move(descriptorSetLayout);
}

void VulkanContext::createPipelineLayout(std::string type_op, std::string ori_op,
                                         [[maybe_unused]] int tensor_count, int params_size) {
    const vk::DescriptorSetLayout descriptorSetLayout = *this->m_descriptor_set_layouts[ori_op];
    std::vector<vk::DescriptorSetLayout> setLayouts{descriptorSetLayout};
    std::vector<vk::PushConstantRange> pushRanges;
    if (params_size > 0) {
        vk::PushConstantRange range{};
        range.stageFlags = vk::ShaderStageFlagBits::eCompute;
        range.offset     = 0;
        range.size       = static_cast<uint32_t>(params_size);
        pushRanges.push_back(range);
    }
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(setLayouts);
    if (!pushRanges.empty()) pipelineLayoutInfo.setPushConstantRanges(pushRanges);

    vk::UniquePipelineLayout pipelineLayout;
    try {
        pipelineLayout = vk::UniquePipelineLayout(
            m_device->createPipelineLayout(pipelineLayoutInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("createPipelineLayout failed: ") + e.what());
    }
    m_pipeline_layouts[type_op] = std::move(pipelineLayout);
}

void VulkanContext::createInstance() {
    if (m_enableValidationLayers && !this->checkLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName("Tensorvia")
        .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
        .setPEngineName("NoEngine")
        .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
        .setApiVersion(VK_API_VERSION_1_4);
    std::vector<const char*> instanceExtensions;
    if (m_enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    vk::InstanceCreateInfo createInfo;
    if (m_enableValidationLayers) {
        createInfo.setPApplicationInfo(&appInfo).setPEnabledLayerNames(m_validationLayers);
    }
    createInfo.setPApplicationInfo(&appInfo).setPEnabledExtensionNames(instanceExtensions);
    // 同 createLogicalDevice：createInstanceUnique 会存好 dispatcher，裸句柄
    // 构造的 UniqueInstance 在 debug 析构时会触发 VULKAN_HPP_ASSERT(m_dispatch)。
    this->m_instance = vk::createInstanceUnique(createInfo);
}

bool VulkanContext::checkLayerSupport() {
    auto availableLayers = vk::enumerateInstanceLayerProperties();
    for (const char* layerName : m_validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) return false;
    }
    return true;
}

bool VulkanContext::checkExtensionSupport() {
    auto availableDeviceExten = m_phydevice.enumerateDeviceExtensionProperties();
    for (const char* extName : m_deviceExtensions) {
        bool found = false;
        for (const auto& prop : availableDeviceExten) {
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

void VulkanContext::setupDebugMessenger() {
    if (!m_enableValidationLayers) return;
    static PFN_vkDebugUtilsMessengerCallbackEXT DebugUtilsMessengerCallback = [](
        [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        [[maybe_unused]] void* pUserData) -> VkBool32 {
        std::println("{}", pCallbackData->pMessage);
        return VK_FALSE;
    };
    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = {},
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = DebugUtilsMessengerCallback,
        .pUserData       = nullptr};
    // Both entry points must be fetched through vkGetInstanceProcAddr: the
    // loader does not export them.
    m_debugMessenger.destroy_fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(static_cast<VkInstance>(*m_instance),
                              "vkDestroyDebugUtilsMessengerEXT"));
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessenger =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(*m_instance, "vkCreateDebugUtilsMessengerEXT"));
    if (vkCreateDebugUtilsMessenger) {
        VkResult result = vkCreateDebugUtilsMessenger(
            *m_instance, &debugUtilsMessengerCreateInfo, nullptr, &m_debugMessenger.handle);
        if (result) {
            std::println("[VulkanBase] ERROR Failed to create a debug messenger! Error code: {}",
                         int32_t(result));
            m_debugMessenger.handle = VK_NULL_HANDLE;
        } else {
            m_debugMessenger.instance = m_instance.get();
        }
    }
}

void VulkanContext::choosePhysicalDevice() {
    auto devices = m_instance->enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    auto find_best_compute_family =
        [](std::vector<vk::QueueFamilyProperties>&& queue_families) -> uint32_t {
        for (uint32_t i = 0; i < queue_families.size(); i++) {
            const auto& family = queue_families[i];
            if ((family.queueFlags & vk::QueueFlagBits::eCompute) &&
                (family.queueFlags & vk::QueueFlagBits::eGraphics)) {
                return i;
            }
        }
        for (uint32_t i = 0; i < queue_families.size(); i++) {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eCompute) return i;
        }
        throw std::runtime_error("No compute queue family found!");
    };
    auto find_best_phydevice = [](std::vector<vk::PhysicalDevice>& devices) -> vk::PhysicalDevice {
        std::vector<vk::PhysicalDeviceType> priority_order = {
            vk::PhysicalDeviceType::eDiscreteGpu, vk::PhysicalDeviceType::eIntegratedGpu,
            vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu,
            vk::PhysicalDeviceType::eOther};
        for (auto type : priority_order) {
            for (auto& device : devices) {
                auto props = device.getProperties();
                if (props.deviceType != type) continue;
                bool has_compute = false;
                for (const auto& family : device.getQueueFamilyProperties()) {
                    if (family.queueFlags & vk::QueueFlagBits::eCompute) {
                        has_compute = true;
                        break;
                    }
                }
                if (!has_compute) continue;
                auto features = device.getFeatures();
                if (!features.shaderInt64) continue;
                auto limits = props.limits;
                if (limits.maxComputeWorkGroupSize[0] < 16 ||
                    limits.maxComputeWorkGroupSize[1] < 16) {
                    continue;
                }
                return device;
            }
        }
        throw std::runtime_error(
            "No suitable physical device found that supports compute shaders!");
    };
    auto phydevice     = find_best_phydevice(devices);
    auto computefamily = find_best_compute_family(phydevice.getQueueFamilyProperties());
    this->m_phydevice  = phydevice;
    this->m_queuefamily = computefamily;
    const auto& wg = phydevice.getProperties().limits.maxComputeWorkGroupCount;
    m_max_workgroup_count = {wg[0], wg[1], wg[2]};
}

void VulkanContext::createLogicalDevice() {
    if (m_queuefamily == UINT32_MAX) {
        throw std::runtime_error(
            "Queue family index not set! Call choosePhysicalDevice first.");
    }
    if (!this->checkExtensionSupport()) {
        throw std::runtime_error("Required Vulkan extensions not supported.");
    }
    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, m_queuefamily, 1, &queuePriority);

    vk::PhysicalDeviceFeatures coreFeatures{};
    coreFeatures.shaderInt64   = VK_TRUE;
    coreFeatures.shaderInt16   = VK_TRUE;
    coreFeatures.shaderFloat64 = VK_TRUE;

    vk::PhysicalDeviceShaderBfloat16FeaturesKHR bf16Features{};
    bf16Features.shaderBFloat16Type = VK_TRUE;

    vk::PhysicalDeviceVulkan11Features vk11Features{};
    vk11Features.storageBuffer16BitAccess = VK_TRUE;
    vk11Features.storagePushConstant16    = VK_TRUE;

    vk::PhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.shaderInt8             = VK_TRUE;
    vk12Features.shaderFloat16          = VK_TRUE;
    vk12Features.storageBuffer8BitAccess = VK_TRUE;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.features = coreFeatures;
    features2.pNext    = &bf16Features;
    bf16Features.pNext = &vk11Features;
    vk11Features.pNext = &vk12Features;
    vk12Features.pNext = nullptr;

    vk::DeviceCreateInfo createInfo{};
    createInfo.setQueueCreateInfos(queueCreateInfo)
        .setEnabledExtensionCount(static_cast<uint32_t>(m_deviceExtensions.size()))
        .setPpEnabledExtensionNames(m_deviceExtensions.data())
        .setPNext(&features2);

    // createDeviceUnique 会把 dispatcher 一起存进 UniqueDevice；裸句柄构造的
    // UniqueDevice 其 ObjectDestroy 里 m_dispatch 是空指针，debug 下析构时
    // vulkan.hpp 的 VULKAN_HPP_ASSERT(m_dispatch) 直接 abort，release 下则是
    // 对空指针解引用的 UB。
    m_device        = m_phydevice.createDeviceUnique(createInfo);
    m_compute_queue = m_device->getQueue(m_queuefamily, 0);
}

void VulkanContext::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(m_queuefamily).setFlags(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
        vk::CommandPoolCreateFlagBits::eTransient);
    try {
        m_command_pool =
            vk::UniqueCommandPool(m_device->createCommandPool(poolInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("Failed to create command pool: ") + e.what());
    }
}

void VulkanContext::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eStorageBuffer, 3000}};
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(2000)
        .setPoolSizes(poolSizes)
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    try {
        m_descriptor_pool =
            vk::UniqueDescriptorPool(m_device->createDescriptorPool(poolInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("Failed to create descriptor pool: ") + e.what());
    }
}

VKBufferGuard VulkanContext::createStagingBuffer(size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(m_submit_mutex);
    vk::Device device = *m_device;
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(bytes).setUsage(vk::BufferUsageFlagBits::eTransferSrc |
                                       vk::BufferUsageFlagBits::eTransferDst);
    vk::Buffer buffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(buffer);
    auto memProps   = m_phydevice.getMemoryProperties();
    const auto want = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent;
    uint32_t index = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & want) == want) {
            index = i;
            break;
        }
    }
    if (index == UINT32_MAX) {
        device.destroyBuffer(buffer);
        throw std::runtime_error("Can't find host-visible, coherent memory type");
    }
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size).setMemoryTypeIndex(index);
    vk::DeviceMemory memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(buffer, memory, 0);
    return VKBufferGuard(device, buffer, memory);
}

void VulkanContext::uploadBuffer(vk::Buffer dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::recursive_mutex> lock(m_submit_mutex);
    vk::Device device = *m_device;
    auto staging = createStagingBuffer(bytes);
    void* mapped = device.mapMemory(staging.memory(), 0, bytes);
    std::memcpy(mapped, src, bytes);
    device.unmapMemory(staging.memory());

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(*m_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    vk::BufferCopy region{};
    region.setSize(bytes);
    cmd.copyBuffer(staging.buffer(), dst, region);
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_compute_queue.submit(submitInfo, fence);
    const vk::Result wait_result = wait_fence(device, fence);
    device.destroyFence(fence);
    device.freeCommandBuffers(*m_command_pool, 1, &cmd);
    check_fence(wait_result, "uploadBuffer: waitForFences", __FILE__, __LINE__);
}

void VulkanContext::downloadBuffer(vk::Buffer src, void* dst, size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard<std::recursive_mutex> lock(m_submit_mutex);
    vk::Device device = *m_device;
    auto staging = createStagingBuffer(bytes);

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(*m_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    vk::BufferCopy region{};
    region.setSize(bytes);
    cmd.copyBuffer(src, staging.buffer(), region);
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = device.createFence({});
    m_compute_queue.submit(submitInfo, fence);
    const vk::Result wait_result = wait_fence(device, fence);
    device.destroyFence(fence);
    device.freeCommandBuffers(*m_command_pool, 1, &cmd);
    check_fence(wait_result, "downloadBuffer: waitForFences", __FILE__, __LINE__);

    void* mapped = device.mapMemory(staging.memory(), 0, bytes);
    std::memcpy(dst, mapped, bytes);
    device.unmapMemory(staging.memory());
}

void VulkanContext::submitCompute(OpType op, DataType dtype,
                                  const std::vector<vk::Buffer>& buffers,
                                  uint32_t gx, uint32_t gy, uint32_t gz,
                                  const void* push_constants, size_t push_size) {
    std::lock_guard<std::recursive_mutex> lock(m_submit_mutex);
    auto ori_op  = op_to_string(op);
    auto type_op = make_pipeline_key(op, dtype);

    // Dispatch limits: fitting in INT32_MAX elements does not make the dispatch
    // grid legal. Validate before allocating anything.
    {
        const uint32_t lim_x = vulkan_detail::clamp_workgroup_count(m_max_workgroup_count[0]);
        const uint32_t lim_y = vulkan_detail::clamp_workgroup_count(m_max_workgroup_count[1]);
        const uint32_t lim_z = vulkan_detail::clamp_workgroup_count(m_max_workgroup_count[2]);
        if (gx > lim_x || gy > lim_y || gz > lim_z)
            throw std::runtime_error(std::format(
                "submitCompute: dispatch grid ({},{},{}) exceeds maxComputeWorkGroupCount "
                "({},{},{}) — tensor is too large for this backend's dispatch grid",
                gx, gy, gz, lim_x, lim_y, lim_z));
    }

    if (!m_pipelines.contains(type_op)) {
        throw std::runtime_error(std::format("can not find pipeline:{}", type_op));
    }
    if (!m_pipeline_layouts.contains(type_op)) {
        throw std::runtime_error(std::format("can not find pipeline layout:{}", type_op));
    }
    if (!m_descriptor_set_layouts.contains(ori_op)) {
        throw std::runtime_error(std::format("can not find descriptor_set_layouts:{}", ori_op));
    }

    vk::Pipeline            pipeline = *m_pipelines[type_op];
    vk::PipelineLayout      layout   = *m_pipeline_layouts[type_op];
    vk::DescriptorSetLayout dsl      = *m_descriptor_set_layouts[ori_op];

    // Validate inputs before allocating a descriptor set: a throw afterwards
    // would leak the set and eventually exhaust the pool.
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i] == VK_NULL_HANDLE) {
            throw std::runtime_error(
                std::format("submitCompute: buffers[{}] is VK_NULL_HANDLE", i));
        }
    }

    std::array<vk::DescriptorSetLayout, 1> setLayouts = {dsl};
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(*m_descriptor_pool)
        .setDescriptorSetCount(1)
        .setSetLayouts(setLayouts);
    auto allocated    = m_device->allocateDescriptorSets(allocInfo);
    vk::DescriptorSet ds = allocated[0];

    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        bufferInfos.emplace_back(buffers[i], 0, VK_WHOLE_SIZE);
    }
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    for (size_t i = 0; i < bufferInfos.size(); ++i) {
        writes.emplace_back(ds, static_cast<uint32_t>(i), 0, 1,
                            vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfos[i],
                            nullptr);
    }
    m_device->updateDescriptorSets(writes, nullptr);

    vk::CommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.setCommandPool(*m_command_pool)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1);
    vk::CommandBuffer cmd = m_device->allocateCommandBuffers(cmdAllocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, ds, nullptr);
    if (push_constants && push_size > 0) {
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                          static_cast<uint32_t>(push_size), push_constants);
    }
    cmd.dispatch(gx, gy, gz);
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = m_device->createFence({});
    m_compute_queue.submit(submitInfo, fence);
    // Keep the result: fence, command buffer and descriptor set must be
    // released whether or not the wait succeeded.
    const vk::Result wait_result = wait_fence(*m_device, fence);

    m_device->destroyFence(fence);
    m_device->freeCommandBuffers(*m_command_pool, 1, &cmd);
    const vk::Result free_result = m_device->freeDescriptorSets(*m_descriptor_pool, 1, &ds);

    check_fence(wait_result, "submitCompute: waitForFences", __FILE__, __LINE__);
    check_result(free_result, "submitCompute: freeDescriptorSets", __FILE__, __LINE__);
}
