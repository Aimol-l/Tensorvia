#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"

class VulkanContext : public ContextImpl {
private:
    // VK_EXT_debug_utils messenger. It is not a vk::Unique* handle: the loader
    // does not export vkDestroyDebugUtilsMessengerEXT, so the destroy function
    // pointer must be fetched through vkGetInstanceProcAddr. It is declared
    // right after m_instance so it is destroyed before it.
    struct DebugMessenger {
        vk::Instance                        instance{};
        VkDebugUtilsMessengerEXT            handle = VK_NULL_HANDLE;
        PFN_vkDestroyDebugUtilsMessengerEXT destroy_fn = nullptr;

        ~DebugMessenger() {
            if (handle != VK_NULL_HANDLE && destroy_fn != nullptr)
                destroy_fn(static_cast<VkInstance>(instance), handle, nullptr);
        }
    };

    // Destruction order matters: members are destroyed in reverse declaration
    // order, and the handles below have hard ordering requirements:
    //   pipeline / layout / descriptor-set layout -> descriptor pool
    //     -> command pool -> device -> messenger -> instance
    vk::UniqueInstance        m_instance;
    DebugMessenger            m_debugMessenger;
    vk::PhysicalDevice        m_phydevice;   // non-owning: dies with m_instance
    vk::UniqueDevice          m_device;

    uint32_t                  m_queuefamily = 0;

    // Device-guaranteed per-axis dispatch maximum. Starts at the spec minimum
    // and is overwritten with the real limit once a physical device is chosen.
    std::array<uint32_t, 3>   m_max_workgroup_count{65535u, 65535u, 65535u};

    vk::Queue                 m_compute_queue;   // non-owning: owned by device

    vk::UniqueCommandPool     m_command_pool;
    vk::UniqueDescriptorPool  m_descriptor_pool;

    // Recursive: VKTensor::init/copy_to/clone use the queue and command pool
    // directly and must serialize with submitCompute. clone takes the lock and
    // then constructs a VKTensor, which takes it again.
    mutable std::recursive_mutex m_submit_mutex;

    // "relu_float32"    -> pipeline
    std::unordered_map<std::string, vk::UniquePipeline>           m_pipelines;
    // "relu_float32"    -> pipeline layout
    std::unordered_map<std::string, vk::UniquePipelineLayout>     m_pipeline_layouts;
    // "relu"            -> descriptor set layout (dtype-independent)
    std::unordered_map<std::string, vk::UniqueDescriptorSetLayout> m_descriptor_set_layouts;

    bool                            m_enableValidationLayers = false;
    const std::vector<const char*>  m_validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char*>  m_deviceExtensions = {
        "VK_KHR_shader_float16_int8",
        "VK_KHR_shader_bfloat16",
        "VK_KHR_8bit_storage",
        "VK_KHR_16bit_storage",
        "VK_KHR_shader_subgroup_extended_types"
    };

    // Builds every shader module and compute pipeline this backend knows about.
    // Called at the end of the constructor, i.e. once per process, lazily on
    // the first Device::VULKAN tensor.
    void registerPipelines();

public:
    VulkanContext();
    ~VulkanContext() override;

    void* ctx_raw_ptr() override {
        throw std::runtime_error("no ctx_raw_ptr!");
    }

    // Raw handles only: callers borrow them, ownership stays here.
    vk::Device         getDevice() const { return *m_device; }
    vk::PhysicalDevice getPhysicalDevice() const { return m_phydevice; }
    vk::CommandPool    commandPool() const { return *m_command_pool; }
    vk::Queue          computeQueue() const { return m_compute_queue; }

    // Serializes queue / command-pool / allocation use with submitCompute.
    std::recursive_mutex& submitMutex() const { return m_submit_mutex; }

    void registerOp(OpType ops, std::vector<DataType>& dtypes,
                    int tensor_count, int params_size);
    void registerOp(OpType ops, DataType dtype,
                    int tensor_count, int params_size);

    // Buffer + its device memory, released together on destruction.
    template <typename T>
    VKBufferGuard createBuffer(const T data);

    // Host-visible staging helpers used by VKTensor's transfer primitives.
    // `bytes` is the payload size; the returned guard owns the staging buffer.
    VKBufferGuard createStagingBuffer(size_t bytes);
    void          uploadBuffer(vk::Buffer dst, const void* src, size_t bytes);
    void          downloadBuffer(vk::Buffer src, void* dst, size_t bytes);

    void submitCompute(OpType op, DataType dtype,
                       const std::vector<vk::Buffer>& buffers,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const void* push_constants, size_t push_size);

private:
    void createInstance();
    void setupDebugMessenger();
    void choosePhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createDescriptorPool();

    bool checkLayerSupport();
    bool checkExtensionSupport();

    void createDescriptorSetLayout(std::string op, int tensor_count, int params_size);
    void createPipelineLayout(std::string type_op, std::string ori_op,
                              int tensor_count, int params_size);
};

// Defined here rather than in the implementation unit so it works for any T
// (the old header declared it and only ConcatParams was instantiated).
template <typename T>
VKBufferGuard VulkanContext::createBuffer(const T data) {
    std::lock_guard<std::recursive_mutex> lock(m_submit_mutex);
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(sizeof(T))
        .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                  vk::BufferUsageFlagBits::eTransferDst |
                  vk::BufferUsageFlagBits::eTransferSrc);
    vk::Buffer buffer = m_device->createBuffer(bufferInfo);
    vk::MemoryRequirements memReqs = m_device->getBufferMemoryRequirements(buffer);
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
    if (!found) {
        m_device->destroyBuffer(buffer);
        throw std::runtime_error("Can't allocate device memory");
    }
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.setAllocationSize(memReqs.size).setMemoryTypeIndex(memoryTypeIndex);
    vk::DeviceMemory memory = m_device->allocateMemory(allocInfo);
    m_device->bindBufferMemory(buffer, memory, 0);

    // From here the guard owns buffer+memory: any throwing path releases both.
    VKBufferGuard guard(*m_device, buffer, memory);
    uploadBuffer(buffer, &data, sizeof(T));
    return guard;
}
