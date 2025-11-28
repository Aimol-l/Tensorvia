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
    auto memProps = m_phydevice.getMemoryProperties();
    std::println("*************************Vulkan Device Info******************");
    std::println("Selected Vulkan Device:{}",std::string_view(props.deviceName));
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        auto heap = memProps.memoryHeaps[i];
        bool isDeviceLocal = (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
        std::println("Memory Heap[{}]: {} MB ({})",
            i,
            heap.size / (1024 * 1024),
            isDeviceLocal ? "DeviceLocal (VRAM)" : "HostVisible (System RAM)"
        );
    }
    // std::println("Global memory size: {}MB",props.limits.maxMemoryAllocationCount/(1024 * 1024));
    std::println("Max Compute Work Group Size: {}",props.limits.maxComputeWorkGroupSize);
    std::println("Max Compute Work Item Size: {}",props.limits.maxComputeWorkGroupCount);
    std::println("Max Compute Work Invocations Size: {}",props.limits.maxComputeWorkGroupInvocations);
    std::println("*************************************************************");
}

void VulkanContext::registerOp(OpType ops,DataType Dtype,int tensor_count, int params_size){
    std::vector<DataType> typs = {Dtype};
    this->registerOp(ops,typs,tensor_count,params_size);
}
// relu_float32   -->  pipeline 
// relu_float32   -->  pipeline_layout
// relu           -->  descriptor_set_layout
void VulkanContext::registerOp(OpType ops,std::vector<DataType>& Dtypes,int tensor_count,int params_size){
    std::string ori_op = op_to_string(ops);
    // 加载一个算子的8个不同类型的shader
    std::vector<std::string> need_types;
    for(auto& type:Dtypes){
        std::string spvFile = std::format("./spv/{}_{}.spv",ori_op,dtype_to_string(type));
        // std::print("[{}] ",spvFile);
        std::ifstream file(spvFile.c_str());
        // 判断算子文件是否存在
        if(!file.good()){
            throw std::runtime_error(std::format("{} not found",spvFile));
        }
        need_types.push_back(std::format("{}",dtype_to_string(type)));
        // 加载shader
        auto spvCode = readSpvFile(spvFile);
        vk::ShaderModuleCreateInfo createInfo;
        createInfo.setCode(spvCode);
        vk::ShaderModule shaderModule = m_device.createShaderModule(createInfo);
        // 配置shader stage
        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
        stageInfo.setModule(shaderModule);
        stageInfo.setPName("main");

        // 创建管线布局
        std::string type_op = std::format("{}_{}",ori_op,dtype_to_string(type));

        this->createDescriptorSetLayout(ori_op,tensor_count,params_size);
        this->createPipelineLayout(type_op,ori_op,tensor_count,params_size);

        // 创建计算管线
        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = this->m_pipeline_layouts[type_op];
        auto result = this->m_device.createComputePipeline(nullptr, pipelineInfo);
        if (result.result != vk::Result::eSuccess) {
            this->m_device.destroyShaderModule(shaderModule);
            throw std::runtime_error("Failed to create compute pipeline!");
        }
        this->m_pipelines[type_op] = result.value;
        this->m_device.destroyShaderModule(shaderModule);
    }
    std::println("{} SPIR-V:{}",ori_op,need_types);
}
void VulkanContext::createDescriptorSetLayout(std::string ori_op, int tensor_count, int params_size) {
    if (m_pipeline_layouts.find(ori_op) != m_pipeline_layouts.end()) return;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(tensor_count);
    for (int i = 0; i < tensor_count; ++i) {
        vk::DescriptorSetLayoutBinding b{};
        b.binding = i;
        b.descriptorType = vk::DescriptorType::eStorageBuffer;
        b.descriptorCount = 1;
        b.stageFlags = vk::ShaderStageFlagBits::eCompute;
        bindings.push_back(b);
    }
    vk::DescriptorSetLayout descriptorSetLayout;
    try {
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);
        descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
    } catch (const vk::SystemError& e) {
        throw std::runtime_error(std::string("createDescriptorSetLayout failed: ") + e.what());
    }
    this->m_descriptor_set_layouts[ori_op] = descriptorSetLayout;
}
void VulkanContext::createPipelineLayout(std::string type_op,std::string ori_op, int tensor_count, int params_size) {
    auto descriptorSetLayout = this->m_descriptor_set_layouts[ori_op];
    // --- 保证容器命名并在作用域内存活 ---
    std::vector<vk::DescriptorSetLayout> setLayouts{ descriptorSetLayout };
    std::vector<vk::PushConstantRange> pushRanges;
    if (params_size > 0) {
        vk::PushConstantRange range{};
        range.stageFlags = vk::ShaderStageFlagBits::eCompute;
        range.offset = 0;
        range.size = static_cast<uint32_t>(params_size);
        pushRanges.push_back(range);
    }
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setSetLayouts(setLayouts);
    if (!pushRanges.empty()) pipelineLayoutInfo.setPushConstantRanges(pushRanges);

    vk::PipelineLayout pipelineLayout;
    try {
        pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);
    } catch (const vk::SystemError& e) {
        m_device.destroyDescriptorSetLayout(descriptorSetLayout);
        throw std::runtime_error(std::string("createPipelineLayout failed: ") + e.what());
    }
    m_pipeline_layouts[type_op] = pipelineLayout;
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
   auto availableDeviceExten = m_phydevice.enumerateDeviceExtensionProperties();
    
    // for (const auto& prop : availableDeviceExten) {
    //     std::println("  {}", std::string_view(prop.extensionName));
    // }
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
    if (!this->checkExtensionSupport()) {
        throw std::runtime_error("Vulkan extension error");
    }

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, m_queuefamily, 1, &queuePriority);

    // 默认开启：  int32,float32
    // 需要主动开启：int8,int16,int64,float16,bfloat16,float64

    // --- Vulkan 1.1 特性 ---
    vk::PhysicalDeviceVulkan11Features vk11{};
    vk11.storageBuffer16BitAccess = VK_TRUE;
    vk11.storagePushConstant16 = VK_TRUE;

    // --- Vulkan 1.2 特性 ---
    vk::PhysicalDeviceVulkan12Features vk12{};
    vk12.shaderInt8 = VK_TRUE;
    vk12.shaderFloat16 = VK_TRUE;
    vk12.storageBuffer8BitAccess = VK_TRUE;

    // --- bfloat16 特性（关键！必须最前面） ---
    vk::PhysicalDeviceShaderBfloat16FeaturesKHR bf16{};
    bf16.shaderBFloat16Type = VK_TRUE;

    // --- core feature ---
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.shaderInt64 = VK_TRUE;
    deviceFeatures.shaderInt16 = VK_TRUE;
    deviceFeatures.shaderFloat64 = VK_TRUE;


    // --- 构造 pNext 链 ---
    bf16.pNext = &vk11; // bfloat16 → vk11
    vk11.pNext = &vk12; // vk11 → vk12

    vk::DeviceCreateInfo createInfo{};
    createInfo
        .setQueueCreateInfos(queueCreateInfo)
        .setPEnabledFeatures(&deviceFeatures)
        .setPNext(&bf16)  // 最前面必须是 bfloat16
        .setEnabledExtensionCount(static_cast<uint32_t>(m_deviceExtensions.size()))
        .setPpEnabledExtensionNames(m_deviceExtensions.data());

    m_device = m_phydevice.createDevice(createInfo);
    m_compute_queue = m_device.getQueue(m_queuefamily, 0);
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


void VulkanContext::printPipeLines(){
    for(auto[key,val]:m_pipelines){
        std::println("compute line: {}",key);
    }
    for(auto[key,val]:m_pipeline_layouts){
        std::println("compute line layout: {}",key);
    }
    for(auto[key,val]:m_descriptor_set_layouts){
        std::println("m_descriptor_set_layouts: {}",key);
    }
}

// relu,float32,buffers,gx,gy,gz,...
void VulkanContext::submitCompute(
    OpType op,
    DataType dtype,
    const std::vector<vk::Buffer>& buffers,
    uint32_t gx, uint32_t gy, uint32_t gz,
    const void* push_constants,
    size_t push_size)
{
    std::lock_guard<std::mutex> lock(m_submit_mutex);
    auto ori_op = op_to_string(op);
    auto type_op = make_pipeline_key(op, dtype); // 正确的 key
    if (!m_pipelines.contains(type_op)) {
        throw std::runtime_error(std::format("can not find pipeline:{}", type_op));
    }
    if (!m_pipeline_layouts.contains(type_op)) {
        throw std::runtime_error(std::format("can not find pipeline layout:{}", type_op));
    }
    if (!m_descriptor_set_layouts.contains(ori_op)) {
        throw std::runtime_error(std::format("can not find descriptor_set_layouts:{}", ori_op));
    }

    vk::Pipeline pipeline = m_pipelines[type_op];      // <- 使用正确的 key
    vk::PipelineLayout layout = m_pipeline_layouts[type_op];
    vk::DescriptorSetLayout dsl = m_descriptor_set_layouts[ori_op];

    // 分配 descriptor set - 使用命名容器，避免临时问题
    std::array<vk::DescriptorSetLayout, 1> setLayouts = { dsl };
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.setDescriptorPool(m_descriptor_pool)
             .setDescriptorSetCount(1)
             .setSetLayouts(setLayouts);
    auto allocated = m_device.allocateDescriptorSets(allocInfo);
    vk::DescriptorSet ds = allocated[0];

    // ----------- 关键：准备持久化的 buffer infos 容器 -----------
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i] == VK_NULL_HANDLE) {
            throw std::runtime_error(std::format("submitCompute: buffers[{}] is VK_NULL_HANDLE", i));
        }
        bufferInfos.emplace_back(buffers[i], 0, VK_WHOLE_SIZE);
    }

    // 写 descriptors 时引用 bufferInfos 中的元素地址，确保 bufferInfos 在 updateDescriptorSets 调用前存活
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(buffers.size());
    for (size_t i = 0; i < bufferInfos.size(); ++i) {
        writes.emplace_back(
            ds,                                  // dstSet
            static_cast<uint32_t>(i),            // dstBinding
            0,                                   // dstArrayElement
            1,                                   // descriptorCount
            vk::DescriptorType::eStorageBuffer,  // descriptorType
            nullptr,                             // pImageInfo
            &bufferInfos[i],                     // pBufferInfo (指向 bufferInfos 中的元素)
            nullptr                              // pTexelBufferView
        );
    }

    m_device.updateDescriptorSets(writes, nullptr);
    // 分配 command buffer
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
   // 提交并等待
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    vk::Fence fence = m_device.createFence({});
    m_compute_queue.submit(submitInfo, fence);
    vk::Result res = m_device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    // 清理
    m_device.destroyFence(fence);
    m_device.freeCommandBuffers(m_command_pool, 1, &cmd);
}