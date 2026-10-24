#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Shared Vulkan vocabulary: the push-constant structs (must match the GLSL
// layouts byte for byte), the buffer+memory RAII holder, and the checks that
// turn a silent vk::Result into an exception.
//
// This is the only partition that includes <vulkan/vulkan.hpp>; the operator
// implementation units get their Vulkan types from here.
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

// ---------------------------------------------------------------------------
// Push constants. Field order and sizeof() must match the corresponding .comp
// shader's push_constant block exactly: push constants are bound by byte
// layout, not by name.
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
    uint32_t prefix_sum[9];   // num + 1
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
    // View origin in elements. Without it a source that is a strided view with
    // a non-zero offset reads from the buffer base instead of the view start.
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

// Broadcast binary ops (add/sub/mul/div vec + comparisons).
// std430-compatible with the GLSL push_constant (sizeof = 120 <= 128):
// flat index -> coordinates -> per-side element offset; broadcast dims have
// stride 0; off_a/off_b/off_o are view offsets in elements.
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
// Buffer + device-memory owner. The old createBuffer returned a bare vk::Buffer
// and dropped its VkDeviceMemory on return: every call leaked one buffer and
// one allocation.
// ---------------------------------------------------------------------------
class VKBufferGuard {
public:
    VKBufferGuard() = default;
    VKBufferGuard(vk::Device device, vk::Buffer buffer, vk::DeviceMemory memory)
        : m_device(device), m_buffer(buffer), m_memory(memory) {}

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

    vk::Buffer       buffer() const { return m_buffer; }
    vk::DeviceMemory memory() const { return m_memory; }

    // Early release. No in-flight command buffer may still reference it.
    void reset() {
        if (m_buffer) {
            m_device.destroyBuffer(m_buffer);
            m_buffer = nullptr;
        }
        if (m_memory) {
            m_device.freeMemory(m_memory);
            m_memory = nullptr;
        }
    }

private:
    void take_from(VKBufferGuard& other) {
        m_device      = other.m_device;
        m_buffer      = other.m_buffer;
        m_memory      = other.m_memory;
        other.m_buffer = nullptr;
        other.m_memory = nullptr;
    }

    vk::Device       m_device{};
    vk::Buffer       m_buffer{};
    vk::DeviceMemory m_memory{};
};

class VulkanContext;

// Repack a strided/offset view into a contiguous buffer. Declared here rather
// than as a RepackImpl specialization so :vulkan_tensor does not have to pull
// in the operator declarations.
void vulkan_repack(const Metadata& meta, vk::Buffer input, vk::Buffer output,
                          std::shared_ptr<VulkanContext> ctx);

namespace vulkan_detail {// This backend's shaders index elements with 32-bit integers. Exceeding
// INT32_MAX elements silently wraps inside the kernel, so the one allocation
// entry point (VKTensor::init) rejects it up front. Every tensor, including
// the base buffer behind a view, goes through there, so this is a
// construction invariant rather than a per-operator check.
//
// TVV_VULKAN_MAX_ELEMS may only tighten the limit (test hook).
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

// Per-axis dispatch limit, normally maxComputeWorkGroupCount. The spec only
// guarantees 65535, while 1D dispatch uses gx = ceil(numel/256).
// TVV_VULKAN_MAX_WORKGROUPS may only tighten it (test hook).
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

inline void check_result(vk::Result r, const char* what, const char* file, int line) {
    if (r == vk::Result::eSuccess) return;
    throw std::runtime_error(std::format("Vulkan error at {}:{}: {} -> {}", file, line,
                                         what, vk::to_string(r)));
}

// Fence wait budget in nanoseconds. Defaults to 10 s; a healthy kernel is
// orders of magnitude faster. The wait is finite because submitCompute holds
// the submit mutex: a lost device would otherwise hang the caller and every
// thread queued behind that lock forever.
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

// Waits with a finite timeout. On timeout the device may still be using the
// fence and command buffer, so waitIdle() is attempted before the caller
// destroys them. vk::SystemError is swallowed: the timeout is already the
// diagnosis, and the caller still has to run its cleanup.
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

// Fence-specific result check: a timeout means the device is most likely lost
// and this fence will never be signalled.
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
