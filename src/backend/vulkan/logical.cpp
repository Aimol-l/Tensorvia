#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"
#include "tensorvia/backend/vulkan/vulkan_ufunc.hpp"

namespace ops {

// 经 ufunc 广播驱动的逻辑比较：dtype 提升、NumPy 广播，输出 INT8。
Tensor EqualImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::Equal, a, b);
}
Tensor NotEqualImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::NotEqual, a, b);
}
Tensor GreaterImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::Greater, a, b);
}
Tensor LessImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::Less, a, b);
}
Tensor GreaterEqualImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::GreaterEqual, a, b);
}
Tensor LessEqualImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::logical_impl(OpType::LessEqual, a, b);
}

// 标量比较：右操作数是单元素 FLOAT32 广播。
Tensor EqualImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::Equal, a, val);
}
Tensor NotEqualImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::NotEqual, a, val);
}
Tensor GreaterImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::Greater, a, val);
}
Tensor LessImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::Less, a, val);
}
Tensor GreaterEqualImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::GreaterEqual, a, val);
}
Tensor LessEqualImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return vk_ufunc::logical_scalar(OpType::LessEqual, a, val);
}

namespace {
// NonZero 的工作组上限。每个工作组在任意长的区间上网格跨步，所以结果数组大小与
// numel 解耦，且始终 <= kMaxGroups。
constexpr uint32_t kMaxGroups = 1024;
constexpr uint32_t kThreads   = 256;
}  // namespace

size_t NonZeroImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto a_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!a_impl) throw std::runtime_error("vulkan NonZero: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan NonZero: missing VulkanContext");

    const int64_t numel = a.numel();
    const uint64_t need = (static_cast<uint64_t>(numel) + kThreads - 1) / kThreads;
    const uint32_t groups =
        static_cast<uint32_t>(std::clamp<uint64_t>(need, 1, kMaxGroups));

    // 按实际组数定大小；着色器绝不会写过该下标。
    Tensor res = Tensor::Fill({static_cast<int64_t>(groups)}, 0.0f, DataType::INT32,
                              Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    if (!res_impl) throw std::runtime_error("vulkan NonZero: expected a VKTensor result");

    ctx_impl->submitCompute(OpType::Nonzero, a.dtype(), {a_impl->buffer(), res_impl->buffer()},
                            groups, 1, 1, &numel, sizeof(int64_t));

    res.to_host();
    size_t count = 0;
    auto res_data = static_cast<int32_t*>(res.data());
    for (uint32_t i = 0; i < groups; ++i) count += static_cast<size_t>(res_data[i]);
    return count;
}

}  // namespace ops
