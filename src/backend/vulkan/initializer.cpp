#include "backend/vulkan/ops/initializer.h"

namespace ops {
Tensor ZerosImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor temp(shape, dtype, Device::VULKAN);
    size_t numel = temp.numel();
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(temp.get_impl()->context());
    return  temp;
}

Tensor OnesImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor temp(shape, dtype, Device::VULKAN);
    size_t numel = temp.numel();
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(temp.get_impl()->context());
    
    return  temp;
}

Tensor FillImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype, float value){
    Tensor temp(shape, dtype, Device::VULKAN);
    size_t numel = temp.numel();
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(temp.get_impl()->context());
    return  temp;
}


Tensor RandomImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype,float min,float max){
    Tensor temp(shape, dtype, Device::VULKAN);
    size_t numel = temp.numel();
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(temp.get_impl()->context());
    return  temp;
}

template struct ZerosImpl<Device::VULKAN>;
template struct OnesImpl<Device::VULKAN>;
template struct FillImpl<Device::VULKAN>;
template struct RandomImpl<Device::VULKAN>;

}