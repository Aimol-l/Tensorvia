#include <cstddef>
#include <memory>

#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"
#include "sycl_island.h"

namespace {
// The queue and the USM allocator are only touched when the first
// Device::SYCL tensor is requested; CPU-only programs never load libsycl.
struct SyclLazyRegistrar {
    SyclLazyRegistrar() {
        register_device_probe(Device::SYCL, [] {
            return DeviceProbe{sycl_c_device_count(), sycl_c_default_device_index()};
        });
        register_backend_initializer(Device::SYCL, [](int index) {
            auto ctx = std::make_shared<SYCLContext>(index);
            register_tensor_impl(DeviceId{Device::SYCL, index},
                                 [ctx](void* ptr, size_t numel, DataType dtype) {
                                     return std::make_shared<SYCLTensor>(ptr, numel, dtype, ctx);
                                 });
        });
    }
} sycl_lazy_registrar;
}  // namespace
