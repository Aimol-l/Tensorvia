#include "core/factory.h"
#include "backend/cpu/cpu_tensor.h"
#include "type_traits.h"

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


#ifdef BACKEND_VULKAN // 需要同时使用cmatmulpu和sycl
#include "backend/vulkan/vulkan_constant.h"
#include "backend/vulkan/vulkan_tensor.h"
#include "backend/vulkan/vulkan_context.h"
struct VulkanRegistrar {
    std::shared_ptr<VulkanContext> ctx =  std::make_shared<VulkanContext>();
    VulkanRegistrar() {
        
        // 注册算子
        ctx->registerOp(OpType::Add,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::AddVec,3,sizeof(int64_t));
        
        ctx->registerOp(OpType::Sub,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::SubVec,3,sizeof(int64_t));

        ctx->registerOp(OpType::Dot,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::DotVec,3,sizeof(int64_t));

        ctx->registerOp(OpType::Relu,1,sizeof(int64_t));
        ctx->registerOp(OpType::Random,1,sizeof(RandomParams));
        ctx->registerOp(OpType::Fill,1,sizeof(ValueParams<float32>));

        // 注册vulkan后端
        register_tensor_impl(Device::VULKAN, [&](void* ptr,int64_t numel, DataType dtype) {
            return std::make_shared<VKTensor>(ptr,numel, dtype,ctx);
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
