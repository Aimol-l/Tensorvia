#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/cuda/cuda_context.hpp"
#include "tensorvia/backend/cuda/cuda_tensor.hpp"

namespace {
// One entry per GPU that has been brought up. It exists only so the exit hook
// below has something to reach: cudaFree from the factory's own static teardown
// happens after the CUDA runtime has unloaded and fails silently, leaving the
// allocators' cached blocks as reported leaks.
std::mutex                                  g_contexts_mtx;
std::vector<std::shared_ptr<CUDAContext>>   g_contexts;

void release_device_cache() {
    std::lock_guard<std::mutex> lock(g_contexts_mtx);
    for (const auto& ctx : g_contexts) {
        if (ctx) ctx->empty_cache();
    }
}

// The driver, streams and allocator are only touched when a Device::CUDA tensor
// is requested; CPU-only programs never load libcuda.
struct CudaLazyRegistrar {
    CudaLazyRegistrar() {
        register_device_probe(Device::CUDA, [] {
            return DeviceProbe{CUDAContext::device_count(),
                               CUDAContext::preferred_device_index()};
        });
        // One context per requested index; the registry remembers which indices
        // have been initialised, so this runs at most once each.
        register_backend_initializer(Device::CUDA, [](int index) {
            auto ctx = std::make_shared<CUDAContext>(index);
            {
                std::lock_guard<std::mutex> lock(g_contexts_mtx);
                g_contexts.push_back(ctx);
            }
            // Registered once, after the runtime initialised (the context ctor
            // did that), so LIFO exit order runs it before CUDA's own teardown.
            static std::once_flag exit_hook;
            std::call_once(exit_hook, [] { (void)std::atexit(&release_device_cache); });

            const DeviceId device{Device::CUDA, index};
            register_tensor_impl(device, [ctx](void* ptr, size_t numel, DataType dtype) {
                return std::make_shared<CUDATensor>(ptr, numel, dtype, ctx);
            });
            // synchronize_all, not synchronize: the device-level contract is
            // "every stream is idle", matching torch.cuda.synchronize().
            register_device_control(device, DeviceControl{
                [ctx] { ctx->synchronize_all(); },
                [ctx] { ctx->empty_cache(); },
            });
        });
    }
} cuda_lazy_registrar;
}  // namespace
