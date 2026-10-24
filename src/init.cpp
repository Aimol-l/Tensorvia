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
        register_tensor_impl(Device::CPU, [this](void* ptr, size_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(this->ctx, ptr, numel, dtype);
        });
    }
} cpu_registrar;
}  // namespace
