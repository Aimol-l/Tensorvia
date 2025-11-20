#include "backend/vulkan/ops/logical.h"

namespace ops {
Tensor EqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);

    return res;
}

Tensor NotEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);

    return res;
}
Tensor GreaterImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);

    return res;
}
Tensor LessImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    return res;
}
Tensor GreaterEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);

    return res;
}
Tensor LessEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);

    return res;
}
size_t NonZeroImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    size_t count = 0;
    return count;
}

template struct EqualImpl<Device::VULKAN>;
template struct NotEqualImpl<Device::VULKAN>;
template struct GreaterImpl<Device::VULKAN>;
template struct LessImpl<Device::VULKAN>;
template struct GreaterEqualImpl<Device::VULKAN>;
template struct LessEqualImpl<Device::VULKAN>;
template struct NonZeroImpl<Device::VULKAN>;
}