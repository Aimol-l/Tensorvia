#include <cstdint>
#include <memory>
#include <stdexcept>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

void ReluImpl<Device::VULKAN>::execute(Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Relu: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Relu: missing VulkanContext");
    int64_t numel = a.numel();
    ctx_impl->submitCompute(OpType::Relu, a.dtype(), {src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

Tensor ReluImpl<Device::VULKAN>::execute(const Tensor& a) {
    Tensor result     = a.clone();
    auto src_impl     = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Relu: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Relu: missing VulkanContext");

    int64_t numel = result.numel();
    ctx_impl->submitCompute(OpType::Relu, result.dtype(), {src_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

void SiluImpl<Device::VULKAN>::execute(Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Silu: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Silu: missing VulkanContext");

    int64_t numel = a.numel();
    ctx_impl->submitCompute(OpType::Silu, a.dtype(),
                            {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

Tensor SiluImpl<Device::VULKAN>::execute(const Tensor& a) {
    // int8/16/32 -> float32；int64 -> float64（int64->float32 不是合法的 Typecast，
    // 与 Softmax 的约定一致）；浮点保持自身 dtype。
    DataType rt = a.dtype();
    if (is_integral_dtype(rt)) rt = DataType::FLOAT32;
    else if (rt == DataType::INT64) rt = DataType::FLOAT64;
    Tensor result = rt == a.dtype() ? a.clone() : a.to_type(rt);

    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!dst_impl) throw std::runtime_error("vulkan Silu: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Silu: missing VulkanContext");

    int64_t numel = result.numel();
    ctx_impl->submitCompute(OpType::Silu, result.dtype(),
                            {dst_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

void TanhImpl<Device::VULKAN>::execute(Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Tanh: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Tanh: missing VulkanContext");

    int64_t numel = a.numel();
    ctx_impl->submitCompute(OpType::Tanh, a.dtype(),
                            {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

Tensor TanhImpl<Device::VULKAN>::execute(const Tensor& a) {
    DataType rt = a.dtype();
    if (is_integral_dtype(rt)) rt = DataType::FLOAT32;
    else if (rt == DataType::INT64) rt = DataType::FLOAT64;
    Tensor result = rt == a.dtype() ? a.clone() : a.to_type(rt);

    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!dst_impl) throw std::runtime_error("vulkan Tanh: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Tanh: missing VulkanContext");

    int64_t numel = result.numel();
    ctx_impl->submitCompute(OpType::Tanh, result.dtype(),
                            {dst_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

void SigmoidImpl<Device::VULKAN>::execute(Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Sigmoid: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Sigmoid: missing VulkanContext");

    int64_t numel = a.numel();
    ctx_impl->submitCompute(OpType::Sigmoid, a.dtype(),
                            {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

Tensor SigmoidImpl<Device::VULKAN>::execute(const Tensor& a) {
    DataType rt = a.dtype();
    if (is_integral_dtype(rt)) rt = DataType::FLOAT32;
    else if (rt == DataType::INT64) rt = DataType::FLOAT64;
    Tensor result = rt == a.dtype() ? a.clone() : a.to_type(rt);

    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!dst_impl) throw std::runtime_error("vulkan Sigmoid: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Sigmoid: missing VulkanContext");

    int64_t numel = result.numel();
    ctx_impl->submitCompute(OpType::Sigmoid, result.dtype(),
                            {dst_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

Tensor SoftmaxImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    const int dims = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dims;
    int32_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape()[i]);

    DataType res_type = a.dtype();
    if (is_integral_dtype(res_type)) {
        res_type = DataType::FLOAT32;
    } else if (res_type == DataType::INT64 || res_type == DataType::FLOAT64) {
        res_type = DataType::FLOAT64;
    }

    Tensor result(a.shape(), res_type, a.device());

    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error("vulkan Softmax: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Softmax: missing VulkanContext");

    SoftmaxParams params{.axis_dim = axis_dim, .outer_dim = outer_dim, .inner_dim = inner_dim};

    // 流水线键是*输入* dtype，而非输出 dtype：整型 softmax 着色器读整型数据、写
    // 浮点结果。
    ctx_impl->submitCompute(OpType::Softmax, a.dtype(),
                            {src_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &params, sizeof(params));
    return result;
}

}  // namespace ops
