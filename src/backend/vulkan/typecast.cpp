#include "backend/vulkan/ops/typecast.h"

namespace ops {
Tensor TypecastImpl<Device::VULKAN>::execute(const Tensor& a, DataType dst_type) {
    Tensor result(a.shape(), dst_type, Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    
    return result;
}
template struct TypecastImpl<Device::VULKAN>;
}