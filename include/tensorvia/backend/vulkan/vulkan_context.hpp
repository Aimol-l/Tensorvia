#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
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
    vk::PhysicalDevice        m_phydevice;   // 不持有：随 m_instance 一起销毁
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

    // VK_KHR_cooperative_matrix 的探测结果。为 true 时 createLogicalDevice 已连同
    // 扩展和特性一起启用；目前只有 f16/bf16 的 MatMul 内核用它。
    bool m_cooperative_matrix = false;
    std::vector<VkCooperativeMatrixPropertiesKHR> m_cooperative_matrix_configs;

    // cooperative matrix 内核按 16 个 32 线程的 subgroup 排布，subgroup 大小不是
    // 32 就不适用。ready 表示内核要求的 16x16x16 配置（A/B 为该 dtype、C/Result
    // 为 f32）存在；active 表示管线确实建成了、正在用它（建失败会回落，见
    // registerOp）。
    uint32_t m_subgroup_size               = 0;
    bool     m_matmul_coopmat_ready_f16    = false;
    bool     m_matmul_coopmat_ready_bf16   = false;
    bool     m_matmul_coopmat_active_f16   = false;
    bool     m_matmul_coopmat_active_bf16  = false;

    // 一次已入队、尚未回收的提交。
    struct InFlight {
        vk::Fence         fence{};
        vk::CommandBuffer cmd{};
        vk::DescriptorSet ds{};      // 没有集合时为空
        uint64_t          seq = 0;
    };

public:
    // 一条提交通道：一条 VkQueue，加上只有它能用的命令池、描述符池与在飞记录。
    // 命令池与描述符池都不是线程安全的，所以“每通道一份”同时也是线程安全的前提。
    //
    // 语义对齐 CUDA 的 stream：每条（线程, 上下文）一条通道，首次使用时从队列池轮转
    // 分配并固定下来；线程数超过通道数时多出来的线程与更早的线程共用一条（提交互斥
    // 保证正确，只降低并发度）。张量记住自己那条通道，释放时交还给它。
    struct QueueChannel {
        vk::Queue                 queue;      // 不持有：归 device 所有
        vk::UniqueCommandPool     command_pool;
        vk::UniqueDescriptorPool  descriptor_pool;
        std::recursive_mutex      mutex;
        std::deque<InFlight>      inflight;
        std::atomic<uint64_t>     submit_seq{0};     // 已提交的最大序号
        std::atomic<uint64_t>     completed_seq{0};  // 已回收的最大序号
    };

private:
    // 等待释放的 buffer；owner 通道的 completed_seq 越过 after_seq 之后才可销毁。
    struct RetiredBuffer {
        vk::Buffer      buffer{};
        VmaAllocation   allocation = nullptr;
        QueueChannel*   owner      = nullptr;
        uint64_t        after_seq  = 0;
    };

    uint64_t                                   m_id = 0;
    std::vector<std::unique_ptr<QueueChannel>> m_queues;
    std::mutex                                 m_queues_mutex;   // 保护线程私有表
    std::atomic<size_t>                        m_next_queue{0};  // 轮转分配游标

    std::mutex                                 m_retired_mutex;
    std::vector<RetiredBuffer>                 m_retired;

    // "relu_float32"    -> pipeline
    std::unordered_map<std::string, vk::UniquePipeline>           m_pipelines;
    // "relu_float32"    -> pipeline layout
    std::unordered_map<std::string, vk::UniquePipelineLayout>     m_pipeline_layouts;
    // "relu"            -> descriptor set layout（与 dtype 无关）
    std::unordered_map<std::string, vk::UniqueDescriptorSetLayout> m_descriptor_set_layouts;

    bool                            m_enableValidationLayers = false;
    const std::vector<const char*>  m_validationLayers = {"VK_LAYER_KHRONOS_validation"};
    // 每个实例都要求的设备扩展定义在 vulkan_context.cpp：设备探测必须应用同一个
    // 过滤器，否则会报告打不开的设备。

    void registerPipelines();

public:
    // 取合适设备列表的第 0 个（见 device_count()）。
    VulkanContext();
    // 取合适设备列表的第 `device_index` 个。每个实例都拥有自己的 VkInstance、
    // VMA 分配器和管线集合，所以打开第二张卡就要再付一遍这些成本。
    explicit VulkanContext(int device_index);
    ~VulkanContext() override;

    int device_index() const { return m_device_index; }

    // VK_KHR_cooperative_matrix 是否可用并已启用：扩展存在、cooperativeMatrix
    // 特性位为真、且设备至少报告一种矩阵配置，三者缺一即为 false。启用后的配置
    // 列表按 M/N/K 与 A/B/C/Result 的元素类型给出，MatMul 的 f16/bf16 内核据此
    // 决定能否走该路径。
    bool supports_cooperative_matrix() const { return m_cooperative_matrix; }
    const std::vector<VkCooperativeMatrixPropertiesKHR>& cooperative_matrix_configs() const {
        return m_cooperative_matrix_configs;
    }

    // 该 dtype 的 MatMul 是否真的走在 cooperative matrix 内核上（探测通过、且
    // 管线建成）。只有 FLOAT16/BFLOAT16 可能为 true；为 false 时走标量内核，
    // 结果相同，只是慢。
    bool matmul_uses_cooperative_matrix(DataType dtype) const;

    // 有多少物理设备通过计算适用性过滤，以及不带索引的调用方应得哪一个。索引空间
    // 是按 choosePhysicalDevice 的优先级排序后的过滤列表，所以 0 号恰好就是当前
    // 默认构造函数会选中的设备。探测使用一个临时实例：物理设备句柄属于那个实例，
    // 只把数量和顺序带出来。
    static int device_count();
    static int preferred_device_index();

    void* ctx_raw_ptr() override {
        throw std::runtime_error("no ctx_raw_ptr!");
    }

    // 只交出裸句柄：调用方借用，所有权仍在这里。
    vk::Device         getDevice() const { return *m_device; }
    vk::PhysicalDevice getPhysicalDevice() const { return m_phydevice; }
    VmaAllocator       allocator() const { return m_vma.handle; }

    // 调用线程的提交通道；首次调用时从队列池领取。之后本线程所有提交、命令缓冲
    // 与描述符分配都走它，对应 CUDA 的 current_stream()。
    QueueChannel& currentQueue();
    // 本上下文预建了多少条提交通道（= min(队列族 queueCount, 8)）。
    size_t queue_channel_count() const { return m_queues.size(); }

    void registerOp(OpType ops, std::vector<DataType>& dtypes,
                    int tensor_count, int params_size);
    void registerOp(OpType ops, DataType dtype,
                    int tensor_count, int params_size);

    VKBufferGuard createDeviceBuffer(size_t bytes, vk::BufferUsageFlags usage);

    // 释放 buffer。提交是异步的，所以这里只把请求排进退役队列：等到 `owner` 通道上
    // 更早的提交都完成（或该通道本就空闲）才真正 vmaDestroyBuffer。`owner` 为空表示
    // 调用线程当前的通道。张量把 buffer 交还给**分配它**的通道。
    void destroyBuffer(vk::Buffer buffer, VmaAllocation allocation,
                       QueueChannel* owner = nullptr) noexcept;

    template <typename T>
    VKBufferGuard createBuffer(const T data);

    VKBufferGuard createStagingBuffer(size_t bytes);
    // 宿主 -> 设备：只入队，不等。
    void uploadBuffer(vk::Buffer dst, const void* src, size_t bytes);
    // 设备 -> 宿主：同步点。先等**调用线程的通道**清空，再拷回来（与 CUDA 的
    // cudaMemcpyAsync + cudaStreamSynchronize 一致）。
    void downloadBuffer(vk::Buffer src, void* dst, size_t bytes);

    // 命令缓冲（以及可选的描述符集）入队到调用线程的通道后立即返回；二者在 fence
    // 有信号后由 reap 回收。
    void submitAsync(vk::CommandBuffer cmd, vk::DescriptorSet ds = {});

    // 设备级同步点：等到**所有线程**的通道上都提交完成，对应
    // torch.cuda.synchronize()。
    void synchronize();

    void submitCompute(OpType op, DataType dtype,
                       const std::vector<vk::Buffer>& buffers,
                       uint32_t gx, uint32_t gy, uint32_t gz,
                       const void* push_constants, size_t push_size);

private:
    // 以下三项都要求持有对应通道的 mutex。
    void reapCompletedLocked(QueueChannel& ch);  // 回收 fence 已有信号的提交
    void finishFrontLocked(QueueChannel& ch);    // 回收队首（其 fence 已有信号）
    void waitLocked(QueueChannel& ch);           // 等到该通道清空并回收

    // 销毁所有已完成提交的退役 buffer；可在任意线程调用。
    void reapRetired();

    void createInstance();
    void setupDebugMessenger();
    void choosePhysicalDevice(int device_index);
    void probeCooperativeMatrix();
    void createLogicalDevice();
    void createVmaAllocator();
    vk::UniqueCommandPool    createCommandPool();
    vk::UniqueDescriptorPool createDescriptorPool();

    bool checkLayerSupport();
    bool checkExtensionSupport();

    void createDescriptorSetLayout(std::string op, int tensor_count, int params_size);
    void createPipelineLayout(std::string type_op, std::string ori_op,
                              int tensor_count, int params_size);
    // 取注册表里 key 对应的 SPIR-V 建计算管线，登记在 type_op 名下。
    void createComputePipeline(const std::string& key, const std::string& type_op);
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
