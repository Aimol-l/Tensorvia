#include <cstddef>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"

namespace {
// The driver, stream and allocator are only touched when the first
// Device::CUDA tensor is requested; CPU-only programs never load libcuda.
struct CudaLazyRegistrar {
    CudaLazyRegistrar() {
        register_backend_initializer(Device::CUDA, [] {
            auto ctx = std::make_shared<CUDAContext>();
            register_tensor_impl(Device::CUDA,
                                 [ctx](void* ptr, size_t numel, DataType dtype) {
                                     return std::make_shared<CUDATensor>(ptr, numel, dtype, ctx);
                                 });
        });
    }
} cuda_lazy_registrar;
}  // namespace
