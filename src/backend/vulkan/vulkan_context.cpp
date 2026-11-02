#include <vulkan/vulkan.hpp>

#include "tensorvia/backend/vulkan/spv_registry.hpp"

#include "_sync_counters.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"

using vulkan_detail::check_fence;
using vulkan_detail::check_result;
using vulkan_detail::wait_fence;

namespace {

// 每条（线程, 上下文）一条提交通道：首次使用时从队列池领取，登记在线程私有表里。
// 上下文 id 永不复用，已销毁上下文残留的线程私有条目不会误配到新上下文。
std::atomic<uint64_t> g_next_context_id{1};
thread_local std::unordered_map<uint64_t, VulkanContext::QueueChannel*> tls_queue;

// 每个上下文最多预建多少条队列。队列数受所选队列族的 queueCount 限制；超过这个
// 数的宿主线程共享 0 号通道（提交互斥保证正确，只降低并发度）。
constexpr size_t kMaxQueues = 8;

// 每个实例都需要的设备扩展。与可用性过滤共享：缺少其中之一的设备无法打开，因此
// 探测时也不能把它计入。
const std::vector<const char*>& required_device_extensions() {
    static const std::vector<const char*> extensions = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_8bit_storage",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_subgroup_extended_types"
    };
    return extensions;
}

uint32_t find_best_compute_family(std::vector<vk::QueueFamilyProperties>&& queue_families) {
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
}

bool is_suitable_physical_device(vk::PhysicalDevice device) {
    bool has_compute = false;
    for (const auto& family : device.getQueueFamilyProperties()) {
        if (family.queueFlags & vk::QueueFlagBits::eCompute) {
            has_compute = true;
            break;
        }
    }
    if (!has_compute) return false;
    if (!device.getFeatures().shaderInt64) return false;
    const auto limits = device.getProperties().limits;
    if (limits.maxComputeWorkGroupSize[0] < 16 ||
        limits.maxComputeWorkGroupSize[1] < 16) {
        return false;
    }
    // 缺少必需扩展的设备无法打开，把它计入只会把失败从探测阶段推迟到首次分配。
    const auto available = device.enumerateDeviceExtensionProperties();
    for (const char* required : required_device_extensions()) {
        bool found = false;
        for (const auto& prop : available) {
            if (std::strcmp(required, prop.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// 所有可用的物理设备，按默认选择一贯使用的优先级顺序分组。本列表中的位置*就是*
// 设备索引：因此索引 0 恰好就是不带索引的默认选择。
std::vector<vk::PhysicalDevice> collect_suitable_devices(vk::Instance instance) {
    const std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    const std::vector<vk::PhysicalDeviceType> priority_order = {
        vk::PhysicalDeviceType::eDiscreteGpu, vk::PhysicalDeviceType::eIntegratedGpu,
        vk::PhysicalDeviceType::eVirtualGpu, vk::PhysicalDeviceType::eCpu,
        vk::PhysicalDeviceType::eOther};
    std::vector<vk::PhysicalDevice> suitable;
    for (auto type : priority_order) {
        for (const auto& device : devices) {
            if (device.getProperties().deviceType != type) continue;
            if (!is_suitable_physical_device(device)) continue;
            suitable.push_back(device);
        }
    }
    return suitable;
}

}  // namespace

int VulkanContext::device_count() {
    try {
        // 不带应用信息和扩展：裸实例就能完成枚举，而且这个实例在任何设备由它创建
        // 之前就被销毁。
        auto instance = vk::createInstanceUnique(vk::InstanceCreateInfo{});
        return static_cast<int>(collect_suitable_devices(*instance).size());
    } catch (const std::exception&) {
        return 0;
    }
}

int VulkanContext::preferred_device_index() {
    return device_count() > 0 ? 0 : -1;
}

VulkanContext::VulkanContext() : VulkanContext(preferred_device_index()) {}

VulkanContext::VulkanContext(int device_index) : m_id(g_next_context_id.fetch_add(1)) {
    this->createInstance();
    this->setupDebugMessenger();
    this->choosePhysicalDevice(device_index);
    this->createLogicalDevice();
    this->createVmaAllocator();
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
    // 所有资源销毁都由 Unique* 成员和 DebugMessenger 负责（声明顺序见对应分块
    // 的文档）。提交是异步的，所以先等队列清空、回收在飞的命令缓冲与退役 buffer，
    // 再销毁设备。
    if (m_device) {
        for (auto& channel : m_queues) {
            try {
                std::lock_guard<std::recursive_mutex> lock(channel->mutex);
                waitLocked(*channel);
            } catch (const vk::SystemError&) {
            }
        }
        try {
            std::lock_guard<std::mutex> lock(m_retired_mutex);
            for (const auto& entry : m_retired) {
                vmaDestroyBuffer(m_vma.handle, static_cast<VkBuffer>(entry.buffer),
                                 entry.allocation);
            }
            m_retired.clear();
        } catch (const vk::SystemError&) {
        }
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
        const std::string key = make_pipeline_key(ops, type);
        // 视图直接指向编译进产物的静态数组：不拷贝。setCode 收的是
        // ArrayProxyNoTemporaries，所以代理得先落成一个具名对象。
        const auto spv = vkspv::get_spv_code(key);
        const vk::ArrayProxyNoTemporaries<const std::uint32_t> code(
            static_cast<std::uint32_t>(spv.words), spv.data);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(code);

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

    registerOp(OpType::Erf, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Sqrt, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Pow, FloatTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Log, FloatTypes, 2, sizeof(ValueParams<float>));
    registerOp(OpType::Clamp, AllTypes, 2, sizeof(ClampParams));

    registerOp(OpType::Relu, AllTypes, 1, sizeof(int64_t));
    registerOp(OpType::Silu, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Tanh, FloatTypes, 2, sizeof(int64_t));
    registerOp(OpType::Sigmoid, FloatTypes, 2, sizeof(int64_t));
    // Softmax 着色器读整型输入、写浮点结果，所以流水线键必须是*输入* dtype；只注册
    // FloatTypes 会让整型输入找不到流水线。
    registerOp(OpType::Softmax, AllTypes, 2, sizeof(SoftmaxParams));

    registerOp(OpType::SumVec, AllTypes, 2, sizeof(ReduceVecParams));

    registerOp(OpType::Mean, AllTypes, 2, sizeof(ReduceVecParams));
    registerOp(OpType::ProdVec, AllTypes, 2, sizeof(ReduceVecParams));
    registerOp(OpType::MinVec, AllTypes, 2, sizeof(ReduceVecParams));
    registerOp(OpType::Argmax, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::Argmin, AllTypes, 2, sizeof(SoftmaxParams));
    registerOp(OpType::MaxVec, AllTypes, 2, sizeof(ReduceVecParams));

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

    // where 的参数放不进 push constant（三套广播步长 + 四个偏移在 8 维下超过 128
    // 字节），与 concat 一样走 binding 0 的 SSBO，因此 params_size 为 0。
    registerOp(OpType::Where, AllTypes, 5, 0);
    registerOp(OpType::IndexSelect, AllTypes, 4, sizeof(IndexAxisParams));
    registerOp(OpType::Gather, AllTypes, 4, sizeof(GatherParams));
    registerOp(OpType::Cumsum, AllTypes, 2, sizeof(CumsumParams));

    registerOp(OpType::SortSmall, AllTypes, 3, sizeof(SortParams));

    registerOp(OpType::SortBlock, AllTypes, 3, sizeof(SortParams));

    // input / weight / bias / output。四个 dtype：卷积只对浮点定义（与 PyTorch 一致）。
    registerOp(OpType::Conv2d, FloatTypes, 4, sizeof(Conv2dParams));
}

void VulkanContext::createDescriptorSetLayout(std::string ori_op, int tensor_count,
                                              [[maybe_unused]] int params_size) {
    // 描述符集布局不携带 dtype 信息，因此只按算子名作键，由所有 dtype 变体共享。
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
    for (const char* extName : required_device_extensions()) {
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
    // 两个入口都必须经 vkGetInstanceProcAddr 获取：loader 并不导出它们。
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

void VulkanContext::choosePhysicalDevice(int device_index) {
    const std::vector<vk::PhysicalDevice> devices = collect_suitable_devices(*m_instance);
    if (devices.empty()) {
        throw std::runtime_error(
            "No suitable physical device found that supports compute shaders!");
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw std::runtime_error(std::format(
            "VulkanContext: device index {} out of range ({} suitable device(s) found)",
            device_index, devices.size()));
    }
    this->m_device_index = device_index;
    auto phydevice      = devices[static_cast<size_t>(device_index)];
    auto computefamily  = find_best_compute_family(phydevice.getQueueFamilyProperties());
    this->m_phydevice   = phydevice;
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
    // 从所选队列族尽量多要几条计算队列，作为每线程提交通道的池子。族的 queueCount
    // 是硬上限；共享同一族的队列之间没有顺序关系，正好对应 CUDA 的 stream。
    const auto families = m_phydevice.getQueueFamilyProperties();
    const uint32_t queue_count = static_cast<uint32_t>(
        std::min<size_t>(std::max(1u, families[m_queuefamily].queueCount), kMaxQueues));
    const std::vector<float> queuePriorities(queue_count, 1.0f);
    vk::DeviceQueueCreateInfo queueCreateInfo({}, m_queuefamily, queue_count,
                                              queuePriorities.data());

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

    const std::vector<const char*>& extensions = required_device_extensions();
    vk::DeviceCreateInfo createInfo{};
    createInfo.setQueueCreateInfos(queueCreateInfo)
        .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
        .setPpEnabledExtensionNames(extensions.data())
        .setPNext(&features2);

    // createDeviceUnique 会把 dispatcher 一起存进 UniqueDevice；裸句柄构造的
    // UniqueDevice 其 ObjectDestroy 里 m_dispatch 是空指针，debug 下析构时
    // vulkan.hpp 的 VULKAN_HPP_ASSERT(m_dispatch) 直接 abort，release 下则是
    // 对空指针解引用的 UB。
    m_device = m_phydevice.createDeviceUnique(createInfo);

    m_queues.reserve(queue_count);
    for (uint32_t i = 0; i < queue_count; ++i) {
        auto channel           = std::make_unique<QueueChannel>();
        channel->queue         = m_device->getQueue(m_queuefamily, i);
        channel->command_pool  = createCommandPool();
        channel->descriptor_pool = createDescriptorPool();
        m_queues.push_back(std::move(channel));
    }
}

VulkanContext::QueueChannel& VulkanContext::currentQueue() {
    const auto it = tls_queue.find(m_id);
    if (it != tls_queue.end()) return *it->second;

    std::lock_guard<std::mutex> lock(m_queues_mutex);
    // 轮转分配：前 N 条线程各占一条，之后按环复用。不记录“已占用”，因此线程结束
    // 也不会把池耗光——并发的线程数不超过通道数时，它们总是分散在不同队列上。
    const size_t pick = m_next_queue.fetch_add(1, std::memory_order_relaxed) % m_queues.size();
    auto* channel     = m_queues[pick].get();
    tls_queue[m_id]   = channel;
    return *channel;
}

void VulkanContext::createVmaAllocator() {
    // 本包以 VMA_DYNAMIC_VULKAN_FUNCTIONS=1、VMA_STATIC_VULKAN_FUNCTIONS=0 构建
    // VMA，所以 VMA 从这两个函数出发自行解析所有入口：缺了它们，vmaCreateAllocator
    // 会解引用空函数指针，而不是干净地失败。
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo info{};
    info.instance         = *m_instance;
    info.physicalDevice   = m_phydevice;
    info.device           = *m_device;
    info.vulkanApiVersion = VK_API_VERSION_1_4;
    info.pVulkanFunctions = &functions;

    vulkan_detail::check_vma(vmaCreateAllocator(&info, &m_vma.handle),
                             "vmaCreateAllocator", __FILE__, __LINE__);
}

// 命令池与描述符池都归单条通道专用：二者都不是线程安全的，按通道分开也就顺手
// 解决了并发提交。
vk::UniqueCommandPool VulkanContext::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(m_queuefamily).setFlags(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
        vk::CommandPoolCreateFlagBits::eTransient);
    try {
        return vk::UniqueCommandPool(m_device->createCommandPool(poolInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("Failed to create command pool: ") + e.what());
    }
}

vk::UniqueDescriptorPool VulkanContext::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eStorageBuffer, 3000}};
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(2000)
        .setPoolSizes(poolSizes)
        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    try {
        return vk::UniqueDescriptorPool(m_device->createDescriptorPool(poolInfo), *m_device);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("Failed to create descriptor pool: ") + e.what());
    }
}

VKBufferGuard VulkanContext::createDeviceBuffer(size_t bytes, vk::BufferUsageFlags usage) {
    // buffer 归属调用线程的通道：释放时要等那条通道上的既有工作完成。
    QueueChannel* owner = &currentQueue();

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(bytes).setUsage(usage);
    // 不加宿主访问标志：这样 VMA_MEMORY_USAGE_AUTO 会优先设备本地内存，正是存储
    // buffer 想要的。
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    const VkBufferCreateInfo ci = static_cast<VkBufferCreateInfo>(bufferInfo);

    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    vulkan_detail::check_vma(
        vmaCreateBuffer(m_vma.handle, &ci, &allocInfo, &buffer, &allocation, nullptr),
        "vmaCreateBuffer(device)", __FILE__, __LINE__);
    return VKBufferGuard(m_vma.handle,
                         [this, owner](vk::Buffer b, VmaAllocation a) {
                             destroyBuffer(b, a, owner);
                         },
                         vk::Buffer(buffer), allocation);
}

void VulkanContext::destroyBuffer(vk::Buffer buffer, VmaAllocation allocation,
                                  QueueChannel* owner) noexcept {
    if (!buffer) return;
    if (owner == nullptr) {
        try {
            owner = &currentQueue();
        } catch (...) {
        }
    }
    if (owner == nullptr) {
        // 连通道都取不到时只剩立即销毁一条路；调用方此时已无法提交新工作。
        vmaDestroyBuffer(m_vma.handle, static_cast<VkBuffer>(buffer), allocation);
        return;
    }

    // 该通道上还没有在飞工作，块不会被任何命令引用，可以立刻销毁——这是绝大多数
    // 临时张量的路径。
    const uint64_t after_seq = owner->submit_seq.load(std::memory_order_relaxed);
    if (owner->completed_seq.load(std::memory_order_acquire) >= after_seq) {
        vmaDestroyBuffer(m_vma.handle, static_cast<VkBuffer>(buffer), allocation);
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(m_retired_mutex);
        m_retired.push_back(RetiredBuffer{buffer, allocation, owner, after_seq});
    } catch (...) {
        // 记不下来就只能先等该通道清空再释放（慢，但不会回收仍在被引用的内存）。
        try {
            std::lock_guard<std::recursive_mutex> lock(owner->mutex);
            waitLocked(*owner);
        } catch (...) {
        }
        vmaDestroyBuffer(m_vma.handle, static_cast<VkBuffer>(buffer), allocation);
    }
}

void VulkanContext::reapRetired() {
    std::lock_guard<std::mutex> lock(m_retired_mutex);
    if (m_retired.empty()) return;
    std::erase_if(m_retired, [this](const RetiredBuffer& entry) {
        if (entry.owner->completed_seq.load(std::memory_order_acquire) < entry.after_seq) {
            return false;
        }
        vmaDestroyBuffer(m_vma.handle, static_cast<VkBuffer>(entry.buffer), entry.allocation);
        return true;
    });
}

void VulkanContext::submitAsync(vk::CommandBuffer cmd, vk::DescriptorSet ds) {
    QueueChannel& ch = currentQueue();
    std::lock_guard<std::recursive_mutex> lock(ch.mutex);
    reapCompletedLocked(ch);
    // 描述符池上限 2000 个集合；积压过多时先等最老的一次，保证池里还有空位。
    constexpr size_t kMaxInFlight = 512;
    if (ch.inflight.size() >= kMaxInFlight) {
        waitLocked(ch);
        tensorvia::sync::note_wait(tensorvia::sync::Wait::Backlog);
    }

    vk::Fence fence = m_device->createFence({});
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    try {
        ch.queue.submit(submitInfo, fence);
    } catch (...) {
        m_device->destroyFence(fence);
        throw;
    }
    const uint64_t seq = ch.submit_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    ch.inflight.push_back(InFlight{fence, cmd, ds, seq});
}

void VulkanContext::finishFrontLocked(QueueChannel& ch) {
    InFlight entry = ch.inflight.front();
    ch.inflight.pop_front();

    m_device->destroyFence(entry.fence);
    m_device->freeCommandBuffers(*ch.command_pool, 1, &entry.cmd);
    if (entry.ds) (void)m_device->freeDescriptorSets(*ch.descriptor_pool, 1, &entry.ds);

    // 该序号及之前的提交都已完成：归这条通道的退役 buffer 可以销毁了。
    ch.completed_seq.store(entry.seq, std::memory_order_release);
    reapRetired();
}

void VulkanContext::reapCompletedLocked(QueueChannel& ch) {
    while (!ch.inflight.empty()) {
        const vk::Result ready = m_device->getFenceStatus(ch.inflight.front().fence);
        if (ready != vk::Result::eSuccess) break;
        finishFrontLocked(ch);
    }
}

void VulkanContext::waitLocked(QueueChannel& ch) {
    while (!ch.inflight.empty()) {
        // 队列有序：等队首的 fence 有信号，就代表排在它之前的都完成了。
        const vk::Result waited = wait_fence(*m_device, ch.inflight.front().fence);
        if (waited != vk::Result::eSuccess) {
            // 超时意味着设备大概率已丢失；先收齐资源再抛，免得泄漏。
            const vk::Result failure = waited;
            finishFrontLocked(ch);
            check_fence(failure, "VulkanContext: waitForFences", __FILE__, __LINE__);
        }
        finishFrontLocked(ch);
    }
}

void VulkanContext::synchronize() {
    // 一次只等一条通道：不同通道之间没有顺序关系，全部等到就是设备级同步。
    for (auto& channel : m_queues) {
        std::lock_guard<std::recursive_mutex> lock(channel->mutex);
        waitLocked(*channel);
        tensorvia::sync::note_wait(tensorvia::sync::Wait::Synchronize);
    }
}

VKBufferGuard VulkanContext::createStagingBuffer(size_t bytes) {
    QueueChannel* owner = &currentQueue();

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(bytes).setUsage(vk::BufferUsageFlagBits::eTransferSrc |
                                       vk::BufferUsageFlagBits::eTransferDst);
    // 随机访问同时覆盖两个方向（上传写、下载读）。MAPPED_BIT 让指针在守卫的生命
    // 周期内保持有效，调用方无需每次传输都 map/unmap。
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

    const VkBufferCreateInfo ci = static_cast<VkBufferCreateInfo>(bufferInfo);

    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo info{};
    vulkan_detail::check_vma(
        vmaCreateBuffer(m_vma.handle, &ci, &allocInfo, &buffer, &allocation, &info),
        "vmaCreateBuffer(staging)", __FILE__, __LINE__);
    if (info.pMappedData == nullptr) {
        vmaDestroyBuffer(m_vma.handle, buffer, allocation);
        throw std::runtime_error("createStagingBuffer: VMA returned an unmapped allocation");
    }
    return VKBufferGuard(m_vma.handle,
                         [this, owner](vk::Buffer b, VmaAllocation a) {
                             destroyBuffer(b, a, owner);
                         },
                         vk::Buffer(buffer), allocation, info.pMappedData);
}

void VulkanContext::uploadBuffer(vk::Buffer dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    QueueChannel& ch = currentQueue();
    std::lock_guard<std::recursive_mutex> lock(ch.mutex);
    vk::Device device = *m_device;
    auto staging = createStagingBuffer(bytes);
    std::memcpy(staging.mapped(), src, bytes);
    // VMA 偏好的相干内存上是空操作；非相干内存时才需要。
    vmaFlushAllocation(m_vma.handle, staging.allocation(), 0, VK_WHOLE_SIZE);

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(*ch.command_pool)
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

    // 只入队。暂存 buffer 离开作用域时经 destroyBuffer 排队，等这次提交完成后才
    // 真正释放。
    submitAsync(cmd);
}

void VulkanContext::downloadBuffer(vk::Buffer src, void* dst, size_t bytes) {
    if (bytes == 0) return;
    QueueChannel& ch = currentQueue();
    std::lock_guard<std::recursive_mutex> lock(ch.mutex);
    vk::Device device = *m_device;
    auto staging = createStagingBuffer(bytes);

    vk::CommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.setCommandPool(*ch.command_pool)
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

    // 命令按队列顺序执行，这次拷贝自然排在本线程之前所有算子之后；等这条通道清空
    // 就拿到了结果。这是 to_host()/item<T>() 的同步点（与 CUDA 只在当前 stream 上
    // 同步一致；跨线程交接的数据需要调用方先 ops::synchronize()）。
    submitAsync(cmd);
    waitLocked(ch);
    tensorvia::sync::note_wait(tensorvia::sync::Wait::Copy);

    vmaInvalidateAllocation(m_vma.handle, staging.allocation(), 0, VK_WHOLE_SIZE);
    std::memcpy(dst, staging.mapped(), bytes);
}

void VulkanContext::submitCompute(OpType op, DataType dtype,
                                  const std::vector<vk::Buffer>& buffers,
                                  uint32_t gx, uint32_t gy, uint32_t gz,
                                  const void* push_constants, size_t push_size) {
    QueueChannel& ch = currentQueue();
    std::lock_guard<std::recursive_mutex> lock(ch.mutex);
    // BOOL 与 INT8 同为单字节布局，且没有自己的着色器：bool buffer 由 INT8 流水线
    // 完成拷贝、比较或填充。宿主侧驱动在进出一律把值规范化为 0/1。
    if (dtype == DataType::BOOL) dtype = DataType::INT8;
    auto ori_op  = op_to_string(op);
    auto type_op = make_pipeline_key(op, dtype);

    // 派发限制：元素数落在 INT32_MAX 内并不保证派发网格合法。先校验再分配任何东西。
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

    // 在分配描述符集之前校验输入：之后再抛异常会泄漏该集合，最终耗尽池。
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i] == VK_NULL_HANDLE) {
            throw std::runtime_error(
                std::format("submitCompute: buffers[{}] is VK_NULL_HANDLE", i));
        }
    }

    std::array<vk::DescriptorSetLayout, 1> setLayouts = {dsl};
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(*ch.descriptor_pool)
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
    cmdAllocInfo.setCommandPool(*ch.command_pool)
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

    // 只入队。命令缓冲与描述符集在 fence 有信号后由 submitAsync 的回收逻辑释放，
    // 所以这里不能立即 freeDescriptorSets。
    submitAsync(cmd, ds);
}
