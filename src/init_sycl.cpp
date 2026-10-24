#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"

namespace {
// The queue and the USM allocator are only touched when the first
// Device::SYCL tensor is requested; CPU-only programs never load libsycl.
struct SyclLazyRegistrar {
    SyclLazyRegistrar() {
        register_backend_initializer(Device::SYCL, [] {
            auto ctx = std::make_shared<SYCLContext>();
            register_tensor_impl(Device::SYCL,
                                 [ctx](void* ptr, size_t numel, DataType dtype) {
                                     return std::make_shared<SYCLTensor>(ptr, numel, dtype, ctx);
                                 });
        });
    }
} sycl_lazy_registrar;
}  // namespace
