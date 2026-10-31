#include <stdexcept>
#include <string>

#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"

namespace {

// 线程私有，与 current device 同理：一个线程修改默认值不得改变另一个线程的张量
// 工厂。
DataType& default_dtype_slot() {
    static thread_local DataType value = DataType::FLOAT32;
    return value;
}

DeviceId& default_device_slot() {
    static thread_local DeviceId value{Device::CPU, 0};
    return value;
}

}  // namespace

DataType default_dtype() { return default_dtype_slot(); }

void set_default_dtype(DataType dtype) {
    if (!is_floating_dtype(dtype))
        throw std::runtime_error(
            "set_default_dtype: only float16, bfloat16, float32 and float64 are allowed");
    default_dtype_slot() = dtype;
}

DeviceId default_device() { return default_device_slot(); }

void set_default_device(DeviceId id) {
    if (id.kind == Device::CPU && id.index != 0)
        throw std::invalid_argument("set_default_device: the CPU backend has only index 0");
    if (id.index < 0)
        throw std::invalid_argument("set_default_device: device index must be non-negative");
    // 立即校验：命名后端没有的设备应在这里失败，而不是等到之后某个线程首次调用
    // 工厂时。
    current_device_index(id.kind);
    if (id.index >= device_probe(id.kind).count)
        throw std::out_of_range(
            "set_default_device: no such " + std::string(device_to_string(id.kind)) +
            " device index " + std::to_string(id.index));
    default_device_slot() = id;
}

DeviceId TensorOptions::resolved_id() const {
    if (device == Device::CPU) return DeviceId{Device::CPU, 0};
    if (device_index >= 0) return DeviceId{device, device_index};
    return DeviceId{device, current_device_index(device)};
}
