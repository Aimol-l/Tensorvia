#pragma once

#include <vulkan/vulkan.hpp>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_mem_alloc.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

// ---------------------------------------------------------------------------
// push constant。字段顺序与 sizeof() 必须与对应 .comp shader 的 push_constant
// 块完全一致：push constant 按字节布局绑定，不按名字。
// ---------------------------------------------------------------------------

template <class T>
    requires(std::same_as<T, int8_t> || std::same_as<T, int16_t> ||
             std::same_as<T, int32_t> || std::same_as<T, int64_t> ||
             std::same_as<T, float16> || std::same_as<T, bfloat16> ||
             std::same_as<T, double> || std::same_as<T, float>)
struct ValueParams {
    T       value;
    int64_t numel;
};

struct RandomParams {
    float    min;
    float    max;
    uint32_t seed;
    int64_t  numel;
};

struct ClampParams {
    float   min;
    float   max;
    int64_t numel;
};

struct MatmulParams {
    uint32_t batch;
    uint32_t N;
    uint32_t K;
    uint32_t M;
};

struct SoftmaxParams {
    int32_t axis_dim;
    int32_t outer_dim;
    int32_t inner_dim;
};

// Mean 的 vec 着色器额外带一个显式分母：两趟全归约的趟 2 要对 B 个部分和求平均，
// 分母是全归约的元素数 N 而不是趟 2 自己的 axis_dim（=B）。0 表示用 axis_dim
// （单趟路径）。
//
// `numel` 是输入的真实元素数，轴循环按它截断。**不能由 outer_dim*axis_dim 推导**：
// 单趟路径下两者相等，但两趟归约趟 1 的几何是 (outer=B, axis=ceil(N/B), inner=1)，
// 那个乘积含补位、比真实元素数大，末段就不会被截断而越界读。
struct ReduceVecParams {
    int32_t axis_dim;
    int32_t outer_dim;  // 输出总数（= outer*inner），派发边界
    int32_t inner_dim;
    int32_t numel;
    int32_t mean_n;
    // 0 = 每线程一个输出（派发 outer_dim/256 个 workgroup）；
    // 1 = 每 workgroup 一个输出、线程沿归约轴分摊（派发 outer_dim 个 workgroup）。
    // 只有 FLOAT32 的归约着色器实现了这一模式，其余 dtype 必须传 0。
    int32_t block_mode;
};

struct Trans2DParams {
    uint32_t rows;
    uint32_t cols;
};

struct TransNDParams {
    int32_t numel;
    int32_t dims;
    int32_t axes[8];
    int32_t in_strides[8];
    int32_t out_strides[8];
};

struct CopyParams {
    uint32_t subnumel;
    uint32_t offset;
};

struct ConcatParams {
    uint32_t num;
    uint32_t axis;
    uint32_t prefix_sum[9];   // num + 1（前缀和）
    uint32_t output_shape[6];
    uint32_t output_strides[6];
    uint32_t input_strides[8][6];
};

struct SliceParams {
    uint32_t ndim;
    uint32_t slice_dims;
    uint32_t input_shape[6];
    uint32_t output_shape[6];
    uint32_t input_strides[6];
    uint32_t slice_starts[6];
    // 视图起点（以元素计）。没有它时，带非零 offset 的跨步视图会从 buffer 基址读，
    // 而不是从视图起点读。
    uint32_t offset;
};

struct RepackParams {
    uint32_t ndim;
    uint32_t dtype_size;
    uint32_t numel;
    uint32_t shape[6];
    uint32_t strides[6];
    uint32_t offset;
};

// 广播二元算子（add/sub/mul/div vec 与比较）。与 GLSL push_constant 的 std430
// 布局兼容（sizeof = 120 <= 128）：扁平下标 -> 坐标 -> 各侧元素偏移；
// 广播维的步长为 0；off_a/off_b/off_o 是以元素计的视图 offset。
struct BcastVecParams {
    int64_t numel;
    int32_t ndim;
    int32_t shape[8];
    int32_t sa[8];
    int32_t sb[8];
    int32_t off_a;
    int32_t off_b;
    int32_t off_o;
};

// ---------------------------------------------------------------------------
// where / index_select / gather / cumsum 的参数。
//
// where 的参数放不进 push constant：三套广播步长 + 四个偏移在 8 维下就超过 128
// 字节。它和 concat 一样放进 binding 0 的 SSBO，字段全部 8 字节，std430 的对齐
// 与 GLSL 结构体逐字节一致。cond 进入着色器前已在宿主上转成 FLOAT64（见
// src/backend/vulkan/index.cpp），因此这里不携带 cond 的 dtype。
// ---------------------------------------------------------------------------
struct WhereParams {
    int64_t numel;
    int64_t ndim;
    int64_t shape[8];
    int64_t sc[8];
    int64_t sa[8];
    int64_t sb[8];
    int64_t off_c;
    int64_t off_a;
    int64_t off_b;
    int64_t off_o;
};

// index_select 与 cumsum 共用的几何：输入视为 outer x axis_len x inner。
// `idx_len` 只对 index_select 有意义，`idx_is_64` 表示 index 的标量宽度。
struct IndexAxisParams {
    int32_t outer;
    int32_t axis_len;
    int32_t inner;
    int32_t idx_len;
    int32_t idx_is_64;
};

// gather：输出形状与 index 相同，非 dim 维的坐标决定输入 base。
struct GatherParams {
    int32_t numel;
    int32_t ndim;
    int32_t dim;
    int32_t idx_is_64;
    int32_t axis_dim;
    int32_t axis_stride;
    int32_t out_shape[8];
    int32_t in_strides[8];
};

struct CumsumParams {
    int32_t outer;
    int32_t axis_len;
    int32_t inner;
};

// sort / topk：输入视为 outer x n x inner，每个 (outer, inner) 列独立排序，输出
// 排好序的前 keep 个（sort 的 keep = n，topk 的 keep = k；descending 对应 topk
// 的 largest）。`padded` 是 n 补齐到的 2 的幂，只有 block 路径的 bitonic 网络用。
// 全部 32 位：Vulkan 后端的元素下标本来就是 32 位。
struct SortParams {
    int32_t outer;
    int32_t n;
    int32_t inner;
    int32_t keep;
    int32_t padded;
    int32_t descending;
};

// conv2d。全部 32 位：Vulkan 后端的元素下标本来就是 32 位，超过 INT32_MAX 的
// 张量在分配期就被拒绝了（vulkan_detail::require_int32_indexable）。
// `has_bias == 0` 时着色器不读 binding 2——那一格绑定的是权重 buffer，仅为满足
// 描述符布局必须非空的要求。
struct Conv2dParams {
    uint32_t batch;
    uint32_t in_ch;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t out_ch;
    uint32_t kh;
    uint32_t kw;
    uint32_t stride;
    uint32_t padding;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t has_bias;
};

// typecast：目标 dtype 的编码。必须与 shader/typecast/*.comp 里的 dst_type
// 分支一一对应——这里刻意不用 DataType 的枚举序号，避免枚举增删悄悄改变语义。
struct TypecastParams {
    int64_t numel;
    int32_t dst_type;
};

// BOOL 与 INT8 共用 storage 布局但写出的值不同（前者规范化为 0/1），所以是两个
// 独立的绑定。
constexpr int32_t kCastDstBool     = 0;
constexpr int32_t kCastDstInt8     = 1;
constexpr int32_t kCastDstInt16    = 2;
constexpr int32_t kCastDstInt32    = 3;
constexpr int32_t kCastDstInt64    = 4;
constexpr int32_t kCastDstFloat16  = 5;
constexpr int32_t kCastDstBFloat16 = 6;
constexpr int32_t kCastDstFloat32  = 7;
constexpr int32_t kCastDstFloat64  = 8;

// typecast 的绑定数：1 个源 + 9 个目标别名绑定（BOOL 与 INT8 各一个）。
constexpr int kTypecastBindings = 10;

// 融合 LayerNorm / RmsNorm：一个 workgroup 一行 [outer, norm_size]。
// 与 GLSL push_constant 的 std430 布局兼容（sizeof = 12）。
struct NormParams {
    int32_t norm_size;
    int32_t outer;
    float   eps;
};

// ---------------------------------------------------------------------------
// 由 VMA 支撑的 buffer 持有者。
//
// Vulkan 只给出少量堆，并要求应用自己做子分配；每个张量单独分配一块
// VkDeviceMemory 还会撞上 maxMemoryAllocationCount（通常 4096）。子分配交给 VMA，
// 因此守卫持有一个 (buffer, allocation) 对，并通过它一起释放。
//
// `mapped()` 只在以 VMA_ALLOCATION_CREATE_MAPPED_BIT 创建的 buffer（中转用 buffer）
// 上非空；它是常驻宿主指针，调用方不得自行释放。
// ---------------------------------------------------------------------------
class VKBufferGuard {
public:
    // 释放钩子。提交是异步的，buffer 可能仍被队列里的命令引用，因此销毁要排到
    // fence 有信号之后（由 VulkanContext::destroyBuffer 负责）；没有钩子时按老规矩
    // 立即 vmaDestroyBuffer。
    using Deleter = std::function<void(vk::Buffer, VmaAllocation)>;

    VKBufferGuard() = default;
    VKBufferGuard(VmaAllocator allocator, Deleter deleter, vk::Buffer buffer,
                  VmaAllocation allocation, void* mapped = nullptr)
        : m_deleter(std::move(deleter)), m_allocator(allocator), m_buffer(buffer),
          m_allocation(allocation), m_mapped(mapped) {}

    ~VKBufferGuard() { reset(); }

    VKBufferGuard(const VKBufferGuard&) = delete;
    VKBufferGuard& operator=(const VKBufferGuard&) = delete;

    VKBufferGuard(VKBufferGuard&& other) noexcept { take_from(other); }
    VKBufferGuard& operator=(VKBufferGuard&& other) noexcept {
        if (this != &other) {
            reset();
            take_from(other);
        }
        return *this;
    }

    vk::Buffer    buffer() const { return m_buffer; }
    VmaAllocation allocation() const { return m_allocation; }
    void*         mapped() const { return m_mapped; }

    // 不销毁任何东西，把所有权交给调用方。VKTensor::init 用它把这对资源搬进
    // 自己的成员。
    void release() {
        m_buffer     = nullptr;
        m_allocation = nullptr;
        m_mapped     = nullptr;
    }

    // 提前释放。有钩子时交给 context 排队（可能仍有 command buffer 在引用它）；
    // 没有钩子时立即销毁，此时不能还有在执行的 command buffer。
    void reset() {
        if (m_buffer) {
            if (m_deleter) {
                m_deleter(m_buffer, m_allocation);
            } else {
                vmaDestroyBuffer(m_allocator, static_cast<VkBuffer>(m_buffer), m_allocation);
            }
            m_buffer     = nullptr;
            m_allocation = nullptr;
            m_mapped     = nullptr;
        }
    }

private:
    void take_from(VKBufferGuard& other) {
        m_deleter    = std::move(other.m_deleter);
        m_allocator  = other.m_allocator;
        m_buffer     = other.m_buffer;
        m_allocation = other.m_allocation;
        m_mapped     = other.m_mapped;
        other.release();
    }

    Deleter       m_deleter;
    VmaAllocator  m_allocator = nullptr;
    vk::Buffer    m_buffer{};
    VmaAllocation m_allocation = nullptr;
    void*         m_mapped     = nullptr;
};

class VulkanContext;

// 把跨步/带 offset 的视图重排进连续 buffer。放在这里而不是做成 RepackImpl 特化，
// 是为了让 :vulkan_tensor 不必引入算子声明。
void vulkan_repack(const Metadata& meta, vk::Buffer input, vk::Buffer output,
                          std::shared_ptr<VulkanContext> ctx);

namespace vulkan_detail {// 本后端的 shader 用 32 位整数做元素下标。元素数超过 INT32_MAX 会在内核里
// 静默回绕，所以唯一的分配入口（VKTensor::init）提前拒绝。所有张量（包括视图背后
// 的基 buffer）都要经过那里，因此这是构造期不变式，而不是逐算子检查。
//
// TVV_VULKAN_MAX_ELEMS 只能收紧上限（测试钩子）。
inline int64_t max_elems() {
    constexpr int64_t kDefault = INT32_MAX;
    if (const char* env = std::getenv("TVV_VULKAN_MAX_ELEMS")) {
        char* end = nullptr;
        const long long v = std::strtoll(env, &end, 10);
        if (end && *end == '\0' && end != env && v > 0 && v < kDefault)
            return static_cast<int64_t>(v);
    }
    return kDefault;
}

// 每轴 dispatch 上限，通常是 maxComputeWorkGroupCount。规范只保证 65535，
// 而 1D dispatch 使用 gx = ceil(numel/256)。TVV_VULKAN_MAX_WORKGROUPS 只能收紧
// 它（测试钩子）。
inline uint32_t clamp_workgroup_count(uint32_t device_limit) {
    if (const char* env = std::getenv("TVV_VULKAN_MAX_WORKGROUPS")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end && *end == '\0' && end != env && v > 0 && v < device_limit)
            return static_cast<uint32_t>(v);
    }
    return device_limit;
}

inline void require_int32_indexable(int64_t numel, const char* what) {
    if (numel <= max_elems()) return;
    throw std::runtime_error(std::format(
        "vulkan {}: {} elements exceed this backend's limit of {} (shader element "
        "indices are 32-bit)",
        what, numel, max_elems()));
}

// 一元/激活内核按线性下标从 buffer 基址读，只有"连续且偏移为 0"的张量满足这个
// 前提。其余视图先物化成连续副本（Tensor::clone 会选 copy_to 或
// clone_as_contiguous，两者都处理 offset）。返回的 storage 只在需要物化时被占用，
// 常见路径上保持空构造。
inline const Tensor& linear_source(const Tensor& a, Tensor& storage) {
    if (a.is_contiguous() && a.offset() == 0) return a;
    storage = a.clone();
    return storage;
}

// 非原地一元/激活的源操作数：dtype 需要提升时（整型输入算浮点结果）先转成 rt，
// 否则直接读原张量。两种情形的结果都由 storage 承接，返回的指针只在调用方持着
// storage 期间有效。
inline const Tensor* unary_source(const Tensor& a, DataType rt, Tensor& storage) {
    if (rt != a.dtype()) {
        storage = a.to_type(rt);
        return &storage;
    }
    return &linear_source(a, storage);
}

// VMA 用 VkResult 而非 vk::Result 报错。
inline void check_vma(VkResult r, const char* what, const char* file, int line) {
    if (r == VK_SUCCESS) return;
    throw std::runtime_error(std::format("VMA error at {}:{}: {} -> VkResult {}", file, line,
                                         what, static_cast<int>(r)));
}

inline void check_result(vk::Result r, const char* what, const char* file, int line) {
    if (r == vk::Result::eSuccess) return;
    throw std::runtime_error(std::format("Vulkan error at {}:{}: {} -> {}", file, line,
                                         what, vk::to_string(r)));
}

// fence 等待预算（纳秒）。默认 10 秒；正常内核快几个数量级。等待必须有上限，因为
// submitCompute 持有提交互斥锁：否则设备丢失会让调用方以及排在这把锁后的所有线程
// 永久挂起。
inline uint64_t fence_wait_timeout_ns() {
    constexpr uint64_t kDefaultMs = 10'000;
    if (const char* env = std::getenv("TVV_VULKAN_FENCE_TIMEOUT_MS")) {
        char* end = nullptr;
        const unsigned long long ms = std::strtoull(env, &end, 10);
        if (end && *end == '\0' && end != env)
            return static_cast<uint64_t>(ms) * 1'000'000ull;
    }
    return kDefaultMs * 1'000'000ull;
}

// 带有限超时的等待。超时后设备可能仍在使用该 fence 和 command buffer，所以在
// 调用方销毁它们之前先尝试 waitIdle()。vk::SystemError 被吞掉：超时本身就是诊断
// 结论，调用方仍需继续执行清理。
inline vk::Result wait_fence(vk::Device device, vk::Fence fence) {
    const vk::Result r = device.waitForFences(fence, VK_TRUE, fence_wait_timeout_ns());
    if (r == vk::Result::eTimeout) {
        try {
            device.waitIdle();
        } catch (const vk::SystemError&) {
        }
    }
    return r;
}

// fence 专用的结果检查：超时意味着设备大概率已丢失，这个 fence 再也不会被通知。
inline void check_fence(vk::Result r, const char* what, const char* file, int line) {
    if (r == vk::Result::eTimeout) {
        throw std::runtime_error(std::format(
            "Vulkan error at {}:{}: {} -> waitForFences timed out after {} ms; the "
            "device is most likely lost (TDR / driver reset / GPU removed) and this "
            "fence will never be signalled",
            file, line, what, fence_wait_timeout_ns() / 1000000ull));
    }
    check_result(r, what, file, line);
}

}  // namespace vulkan_detail
