#include <cstddef>
#include <cstdlib>
#include <memory>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"

namespace {
// Process-lifetime holder for the context the factory keeps. It exists only so
// the exit hook below has something to reach: cudaFree from the factory's own
// static teardown happens after the CUDA runtime has unloaded and fails
// silently, leaving the allocator's cached blocks as reported leaks.
std::shared_ptr<CUDAContext> g_context;

void release_device_cache() {
    if (g_context) g_context->empty_cache();
}

// The driver, stream and allocator are only touched when the first
// Device::CUDA tensor is requested; CPU-only programs never load libcuda.
struct CudaLazyRegistrar {
    CudaLazyRegistrar() {
        register_backend_initializer(Device::CUDA, [] {
            auto ctx = std::make_shared<CUDAContext>();
            // Registered after the runtime initialised (the context ctor did
            // that), so LIFO exit order runs this before CUDA's own teardown.
            if (!g_context) {
                g_context = ctx;
                (void)std::atexit(&release_device_cache);
            }
            register_tensor_impl(Device::CUDA,
                                 [ctx](void* ptr, size_t numel, DataType dtype) {
                                     return std::make_shared<CUDATensor>(ptr, numel, dtype, ctx);
                                 });
        });
    }
} cuda_lazy_registrar;
}  // namespace
