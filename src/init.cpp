#include "core/factory.h"
#include "backend/cpu/cpu_tensor.h"


#ifdef BACKEND_CPU
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,numel, dtype);
        });
    }
} cpu_registrar;
#endif

#ifdef BACKEND_CUDA
#include "backend/cuda/cuda_tensor.h"
#include "backend/cuda/cuda_context.h"
struct CUDARegistrar {
    std::shared_ptr<CUDAContext> ctx =  std::make_shared<CUDAContext>();
    CUDARegistrar() {
        register_tensor_impl(Device::CUDA, [&](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<CUDATensor>(ptr,numel, dtype,ctx);
        });
    }
}cuda_registrar;
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,numel, dtype);
        });
    }
}cpu_registrar;
#endif

#ifdef BACKEND_SYCL // 需要同时使用cpu和sycl
#include "backend/sycl/sycl_tensor.h"
#include "backend/sycl/sycl_context.h"
struct SYCLRegistrar {
    std::shared_ptr<SYCLContext> ctx =  std::make_shared<SYCLContext>();
    SYCLRegistrar() {
        register_tensor_impl(Device::SYCL, [&](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<SYCLTensor>(ptr,numel, dtype,ctx);
        });
    }
} sycl_registrar;
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,numel, dtype);
        });
    }
} cpu_registrar;
#endif


#ifdef BACKEND_VULKAN // 需要同时使用cpu和sycl
#include "backend/vulkan/vulkan_tensor.h"
#include "backend/vulkan/vulkan_context.h"
struct VulkanRegistrar {
    std::shared_ptr<VulkanContext> ctx =  std::make_shared<VulkanContext>();
    VulkanRegistrar() {
        // 注册算子

        // 注册vulkan后端
        register_tensor_impl(Device::SYCL, [&](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<VulkanTensor>(ptr,numel, dtype,ctx);
        });
    }
} vulkan_registrar;
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,numel, dtype);
        });
    }
} cpu_registrar;

#endif
