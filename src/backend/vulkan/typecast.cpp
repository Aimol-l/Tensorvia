#include <cstddef>
#include <print>
#include <stdexcept>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"

namespace ops {

// 不是原生内核：经宿主机往返，在宿主上转换。
Tensor TypecastImpl<Device::VULKAN>::execute(const Tensor& a, DataType dst_type) {
    Tensor result = a.clone();
    result.to_host();
    result.to_device(Device::VULKAN);
    result.to_type_(dst_type);
    result.to_device(Device::VULKAN);
    return result;
}

void TypecastImpl<Device::VULKAN>::execute(Tensor& a, DataType dst_type) {
    a.to_host();
    a.to_type_(dst_type);
    a.to_device(Device::VULKAN);
#ifndef NDEBUG
    std::println(stderr,"warning: Vulkan Typecast is implemented as a host round-trip (no compute ""shader)");
#endif
}

}  // namespace ops
