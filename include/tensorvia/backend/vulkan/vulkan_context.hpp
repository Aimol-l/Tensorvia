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
    struct DebugMessenger {
        vk::Instance                        instance{};
        VkDebugUtilsMessengerEXT            handle = VK_NULL_HANDLE;
        PFN_vkDestroyDebugUtilsMessengerEXT destroy_fn = nullptr;

        ~DebugMessenger() {
            if (handle != VK_NULL_HANDLE && destroy_fn != nullptr)
                destroy_fn(static_cast<VkInstance>(instance), handle, nullptr);
        }
    };

    vk::UniqueInstance        m_instance;
    DebugMessenger            m_debugMessenger;
    vk::PhysicalDevice        m_phydevice;   // non-owning: dies with m_instance
    vk::UniqueDevice          m_device;

    struct VmaOwner {
        VmaAllocator handle = nullptr;
        ~VmaOwner() {
            if (handle) vmaDestroyAllocator(handle);
        }
    };
    VmaOwner                  m_vma;

    uint32_t                  m_queuefamily = 0;
    int                       m_device_index = 0;

    std::array<uint32_t, 3>   m_max_workgroup_count{65535u, 65535u, 65535u};

    vk::Queue                 m_compute_queue;   // non-owning: owned by device

    vk::UniqueCommandPool     m_command_pool;
    vk::UniqueDescriptorPool  m_descriptor_pool;

    mutable std::recursive_mutex m_submit_mutex;

    // "relu_float32"    -> pipeline
    std::unordered_map<std::string, vk::UniquePipeline>           m_pipelines;
    // "relu_float32"    -> pipeline layout
    std::unordered_map<std::string, vk::UniquePipelineLayout>     m_pipeline_layouts;
    // "relu"            -> descriptor set layout (dtype-independent)
    std::unordered_map<std::string, vk::UniqueDescriptorSetLayout> m_descriptor_set_layouts;

    bool                            m_enableValidationLayers = false;
    const std::vector<const char*>  m_validationLayers = {"VK_LAYER_KHRONOS_validation"};
    // The device extensions every instance requires live in
    // vulkan_context.cpp: the device probe has to apply the same filter, or it
    // would report devices that cannot be opened.

    void registerPipelines();

public:
    // Picks index 0 of the suitable-device list (see device_count()).
    VulkanContext();
    // Picks position `device_index` of the suitable-device list. Every instance
    // owns its own VkInstance, VMA allocator and pipeline set, so opening a
    // second GPU costs a second copy of all of that.
    explicit VulkanContext(int device_index);
    ~VulkanContext() override;

    int device_index() const { return m_device_index; }

    // How many physical devices pass the compute suitability filter, and which
    // one a caller that names no index should get. The index space is the
    // filtered list in the priority order choosePhysicalDevice uses, so index 0
    // is exactly the device today's default constructor picks. The probe uses a
    // throwaway instance: the physical-device handles belong to that instance,
    // only the count and the ordering are carried over.
    static int device_count();
    static int preferred_device_index();

    void* ctx_raw_ptr() override {
        throw std::runtime_error("no ctx_raw_ptr!");
    }

    // Raw handles only: callers borrow them, ownership stays here.
    vk::Device         getDevice() const { return *m_device; }
    vk::PhysicalDevice getPhysicalDevice() const { return m_phydevice; }
    vk::CommandPool    commandPool() const { return *m_command_pool; }
    vk::Queue          computeQueue() const { return m_compute_queue; }
    VmaAllocator       allocator() const { return m_vma.handle; }

    // Serializes queue / command-pool / allocation use with submitCompute.
    std::recursive_mutex& submitMutex() const { return m_submit_mutex; }

    void registerOp(OpType ops, std::vector<DataType>& dtypes,
                    int tensor_count, int params_size);
    void registerOp(OpType ops, DataType dtype,
                    int tensor_count, int params_size);

    VKBufferGuard createDeviceBuffer(size_t bytes, vk::BufferUsageFlags usage);

    void destroyBuffer(vk::Buffer buffer, VmaAllocation allocation) noexcept;

    template <typename T>
    VKBufferGuard createBuffer(const T data);

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
    void choosePhysicalDevice(int device_index);
    void createLogicalDevice();
    void createVmaAllocator();
    void createCommandPool();
    void createDescriptorPool();

    bool checkLayerSupport();
    bool checkExtensionSupport();

    void createDescriptorSetLayout(std::string op, int tensor_count, int params_size);
    void createPipelineLayout(std::string type_op, std::string ori_op,
                              int tensor_count, int params_size);
};

template <typename T>
VKBufferGuard VulkanContext::createBuffer(const T data) {
    VKBufferGuard guard = createDeviceBuffer(
        sizeof(T), vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eTransferDst |
                       vk::BufferUsageFlagBits::eTransferSrc);
    uploadBuffer(guard.buffer(), &data, sizeof(T));
    return guard;
}
