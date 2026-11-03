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
    
std::mutex                                  g_contexts_mtx;
std::vector<std::shared_ptr<CUDAContext>>   g_contexts;

void release_device_cache() {
    std::lock_guard<std::mutex> lock(g_contexts_mtx);
    for (const auto& ctx : g_contexts) {
        if (ctx) ctx->empty_cache();
    }
}

// 只有在请求 Device::CUDA 张量时才会触及驱动、stream 与分配器；纯 CPU 程序从不
// 加载 libcuda。
struct CudaLazyRegistrar {
    CudaLazyRegistrar() {
        register_device_probe(Device::CUDA, [] {
            return DeviceProbe{CUDAContext::device_count(),
                               CUDAContext::preferred_device_index()};
        });
        // 每个被请求的索引一个上下文；注册表记得哪些索引已初始化，所以每个最多运行一次。
        register_backend_initializer(Device::CUDA, [](int index) {
            auto ctx = std::make_shared<CUDAContext>(index);
            {
                std::lock_guard<std::mutex> lock(g_contexts_mtx);
                g_contexts.push_back(ctx);
            }
            // 在运行时初始化之后才注册一次（由上下文构造函数完成初始化），因此
            // LIFO 退出顺序会让它在 CUDA 自身拆卸之前运行。
            static std::once_flag exit_hook;
            std::call_once(exit_hook, [] { (void)std::atexit(&release_device_cache); });

            const DeviceId device{Device::CUDA, index};
            register_tensor_impl(device, [ctx](void* ptr, size_t numel, DataType dtype) {
                return std::make_shared<CUDATensor>(ptr, numel, dtype, ctx);
            });
            // 用 synchronize_all 而非 synchronize：设备级契约是“所有 stream 空闲”，
            // 与 torch.cuda.synchronize() 一致。
            register_device_control(device, DeviceControl{
                [ctx] { ctx->synchronize_all(); },
                [ctx] { ctx->empty_cache(); },
            });
        });
    }
} cuda_lazy_registrar;
}  // namespace
