#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/cpu_tensor.hpp"
#include "tensorvia/cpu_context.hpp"

namespace {
struct CPURegistrar {
    std::shared_ptr<CPUContext> ctx = std::make_shared<CPUContext>();

    CPURegistrar() {
        // 宿主就是一个实例：索引 0 是唯一存在的 CPU。
        const DeviceId cpu{Device::CPU, 0};
        register_tensor_impl(cpu, [this](void* ptr, size_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(this->ctx, ptr, numel, dtype);
        });
        // 调用返回时宿主必然已完成，而且它不替设备缓存任何东西：两者都是空
        // 操作，因此通用代码可以对拿到的任何设备调用它们。
        register_device_control(cpu, DeviceControl{[] {}, [] {}});
        register_device_probe(Device::CPU, [] { return DeviceProbe{1, 0}; });
    }
} cpu_registrar;
}  // namespace
