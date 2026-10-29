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
        // The host is one instance: index 0 is the only CPU that exists.
        const DeviceId cpu{Device::CPU, 0};
        register_tensor_impl(cpu, [this](void* ptr, size_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(this->ctx, ptr, numel, dtype);
        });
        // The host is always finished when a call returns, and it caches nothing
        // on the device's behalf: both are no-ops, so generic code can call them
        // on whatever device it is given.
        register_device_control(cpu, DeviceControl{[] {}, [] {}});
        register_device_probe(Device::CPU, [] { return DeviceProbe{1, 0}; });
    }
} cpu_registrar;
}  // namespace
