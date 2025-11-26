#include "backend/vulkan/ops/typecast.h"
#include "ops.h"

namespace ops {
Tensor TypecastImpl<Device::VULKAN>::execute(const Tensor& a, DataType dst_type) {
    Tensor result = a.clone();
    result.to_host();
    result.to_device();
    result.to_type(dst_type);
    result.to_device();
    return result;
}

void TypecastImpl<Device::VULKAN>::execute(Tensor& a, DataType dst_type) {
    // 非原生实现
    a.to_host();
    a.to_type(dst_type);
    a.to_device();
    LOG_WARN("TypecastImpl<Device::VULKAN>::execute() is not implemented by original author");
}

template struct TypecastImpl<Device::VULKAN>;

}