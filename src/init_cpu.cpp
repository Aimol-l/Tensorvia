#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/cpu/cpu_tensor.hpp"
#include "tensorvia/backend/cpu/cpu_context.hpp"

namespace {
struct CPURegistrar {
    std::shared_ptr<CPUContext> ctx = std::make_shared<CPUContext>();

    CPURegistrar() {
        const DeviceId cpu{Device::CPU, 0};
        register_tensor_impl(cpu, [this](void* ptr, size_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(this->ctx, ptr, numel, dtype);
        });
        register_device_control(cpu, DeviceControl{[] {}, [] {}});
        register_device_probe(Device::CPU, [] { return DeviceProbe{1, 0}; });
    }
} cpu_registrar;
}  // namespace
