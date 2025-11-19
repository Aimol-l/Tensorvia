#include "vulkan_context.h"



VulkanContext::VulkanContext(){
    this->createInstance();
    this->setupDebugMessenger();
    this->choosePhysicalDevice();       // 按优先级选择设备：独显 > 集显 > 其他
    this->createLogicalDevice();
    this->createCommandPool();          // 创建命令池
    this->createDescriptorPool();       // 创建描述符池

    // 打印设备信息
    auto props = m_phydevice.getProperties();
    std::println("*************************Vulkan Device Info******************");
    std::println("Selected Vulkan Device:{}",props.deviceName);
    std::println("Global memory size: {}MB",props.limits.maxMemoryAllocationCount/(1024 * 1024));
    std::println("Max Compute Work Group Size: [{}]",props.limits.maxComputeWorkGroupSize);
    std::println("Max Compute Work Item Size: [{}]",props.limits.maxComputeWorkGroupCount);
    std::println("Max Compute Work Invocations Size: [{}]",props.limits.maxComputeWorkGroupInvocations);
    std::println("***********************************************************");
}

void VulkanContext::registerOp(OpType ops,int tensor_count,int params_size){
    static constexpr DataType REQUIREDTYPE[] = {
        DataType::INT8, DataType::INT16, DataType::INT32, DataType::INT64,
        DataType::FLOAT16, DataType::FLOAT32, DataType::FLOAT64,
        DataType::BFLOAT16
    };
    std::string op = op_to_string(ops);
    // 创建 pipelineLayout
    this->createPipelineLayout(op,tensor_count,params_size);
    // 加载一个算子的8个不同类型的shader
    for(auto& type:REQUIREDTYPE){
        std::string spvFile = std::format("{}_{}.spv",op,type);
        std::ifstream file(spvFile.c_str());
        // 判断算子文件是否存在
        if(!file.good()){
            throw std::runtime_error(std::format("{} not found",spvFile));
        }
        // 1. 加载shader
        auto spvCode = readSpvFile(spvFile);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(spvCode);
        vk::ShaderModule shaderModule = m_device.createShaderModule(createInfo);
        // 2. 配置shader stage
        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        stageInfo.setModule(shaderModule);
        stageInfo.setPName("main");
        // 3. 创建计算管线
        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = this->m_pipeline_layouts[op];
        auto result = this->m_device.createComputePipeline(nullptr, pipelineInfo);
        if (result.result != vk::Result::eSuccess) {
            this->m_device.destroyShaderModule(shaderModule);
            throw std::runtime_error("Failed to create compute pipeline!");
        }
        this->m_pipelines[spvFile] = result.value;
        this->m_device.destroyShaderModule(shaderModule);
    }
}

void VulkanContext::createInstance(){
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
    createInfo.setPApplicationInfo(&appInfo)
        .setPEnabledLayerNames(m_validationLayers)
        .setPEnabledExtensionNames(instanceExtensions);
    this->m_instance = vk::createInstance(createInfo);
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
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

bool VulkanContext::checkExtensionSupport() {
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

void VulkanContext::setupDebugMessenger() {
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

void VulkanContext::choosePhysicalDevice(){
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

void VulkanContext::createLogicalDevice() {
    if (m_queuefamily == UINT32_MAX) {
        throw std::runtime_error("Queue family index not set! Call choosePhysicalDevice first.");
    }
    if(!this->checkExtensionSupport()){
        throw std::runtime_error("vulkan exetension error");
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

void VulkanContext::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(m_queuefamily)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient);
    
    try {
        m_command_pool = m_device.createCommandPool(poolInfo);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("Failed to create command pool: ") + e.what());
    }
}
void VulkanContext::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eStorageBuffer, 3000}      // 存储缓冲区 - 用于所有张量数据
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

std::vector<uint32_t> VulkanContext::readSpvFile(const std::string& filename) {
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

void VulkanContext::createPipelineLayout(std::string op,int tensor_count,int params_size) {
    // 如果已经创建过就直接返回（避免重复创建）
    if (m_pipeline_layouts.find(op) != m_pipeline_layouts.end()) return;

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
    vk::DescriptorSetLayout descriptorSetLayout;
    try {
        descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("createDescriptorSetLayout failed: ") + e.what());
    }
    // 保存 descriptor set layout 以便后续 allocate 使用与销毁管理
    this->m_descriptor_set_layouts[op] = descriptorSetLayout;

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
    }else {
        pipelineLayoutInfo.setPushConstantRangeCount(0)
                          .setPPushConstantRanges(nullptr);
    }
    pipelineLayoutInfo.setSetLayoutCount(1)
                    .setPSetLayouts(&descriptorSetLayout);
    vk::PipelineLayout pipelineLayout;
    try {
        pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);
    } catch (const vk::SystemError& e) {
        // 失败时销毁先前创建的 descriptorSetLayout 并抛出
        m_device.destroyDescriptorSetLayout(descriptorSetLayout);
        throw std::runtime_error(std::string("createPipelineLayout failed: ") + e.what());
    }
    this->m_pipeline_layouts[op] = pipelineLayout;
}
void VulkanContext::submitCompute(
    const std::string& op_name,
    const std::vector<vk::Buffer>& buffers,
    uint32_t gx, uint32_t gy, uint32_t gz,
    const void* push_constants,
    size_t push_size)
{
    // 1. 查表
    auto pipeline_it = m_pipelines.find(op_name);
    auto layout_it   = m_pipeline_layouts.find(op_name);
    auto dsl_it      = m_descriptor_set_layouts.find(op_name);
    if (pipeline_it == m_pipelines.end() ||
        layout_it == m_pipeline_layouts.end() ||
        dsl_it == m_descriptor_set_layouts.end()) {
        throw std::runtime_error("Operator not registered: " + op_name);
    }

    vk::Pipeline pipeline = pipeline_it->second;
    vk::PipelineLayout layout = layout_it->second;
    vk::DescriptorSetLayout dsl = dsl_it->second;

    // 2. 👉 重置 descriptor pool（关键！释放上一次所有 descriptor sets）
    m_device.resetDescriptorPool(m_descriptor_pool);

    // 3. 分配 descriptor set
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(m_descriptor_pool)
             .setDescriptorSetCount(1)
             .setSetLayouts(dsl);
    vk::DescriptorSet ds = m_device.allocateDescriptorSets(allocInfo)[0];

    // 4. 写入 buffers (binding = index)
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        vk::DescriptorBufferInfo bufferInfo{ buffers[i], 0, VK_WHOLE_SIZE };
        writes.emplace_back(
            ds,
            static_cast<uint32_t>(i), // binding
            0,
            1,
            vk::DescriptorType::eStorageBuffer,
            nullptr,
            &bufferInfo
        );
    }
    m_device.updateDescriptorSets(writes, nullptr);

    // 5. 分配 command buffer
    vk::CommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.setCommandPool(m_command_pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);
    vk::CommandBuffer cmd = m_device.allocateCommandBuffers(cmdAllocInfo)[0];

    // 6. Record
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd.begin(beginInfo);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, ds, nullptr);
    if (push_constants && push_size > 0) {
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, static_cast<uint32_t>(push_size), push_constants);
    }
    cmd.dispatch(gx, gy, gz);
    cmd.end();

    // 7. Submit & wait
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);

    vk::Fence fence = m_device.createFence({});
    m_compute_queue.submit(submitInfo, fence);
    m_device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 8. Cleanup
    m_device.destroyFence(fence);
    m_device.freeCommandBuffers(m_command_pool, 1, &cmd);
    // descriptor set 会被 resetDescriptorPool 自动回收，无需手动释放
}