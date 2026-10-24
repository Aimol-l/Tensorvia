#include <cstddef>
#include <print>
#include <stdexcept>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"

namespace ops {

// Not a native kernel: round-trip through the host and convert there.
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
    std::println(stderr,
                 "warning: Vulkan Typecast is implemented as a host round-trip (no compute "
                 "shader)");
}

}  // namespace ops
