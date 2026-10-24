#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/vulkan/vulkan_ops.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"
#include "interface/backend/vulkan/vulkan_ufunc.hpp"

namespace ops {

// Logical comparison through the ufunc broadcast driver: dtype promotion,
// NumPy broadcasting and output INT8.
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

// Scalar comparison: the right operand is a one-element FLOAT32 broadcast.
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
// NonZero workgroup cap. Each workgroup grid-strides over an arbitrarily long
// interval, so the result array size is decoupled from numel and is always
// <= kMaxGroups.
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

    // Sized by the real group count; the shader never writes past that index.
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
