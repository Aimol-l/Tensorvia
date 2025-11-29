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
        static std::vector<DataType> AllTypes = {
            DataType::INT8,
            DataType::INT16,
            DataType::INT32,
            DataType::INT64,
            // DataType::FLOAT16,
            // DataType::BFLOAT16,
            DataType::FLOAT32,
            DataType::FLOAT64
        };
        static std::vector<DataType> FloatTypes = {
            // DataType::FLOAT16,
            // DataType::BFLOAT16,
            DataType::FLOAT32,
            DataType::FLOAT64
        };

        // 注册算子
        ctx->registerOp(OpType::Add,AllTypes,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::AddVec,AllTypes,3,sizeof(int64_t));
        
        ctx->registerOp(OpType::Sub,AllTypes,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::SubVec,AllTypes,3,sizeof(int64_t));

        ctx->registerOp(OpType::Dot,AllTypes,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::DotVec,AllTypes,3,sizeof(int64_t));

        ctx->registerOp(OpType::Div,AllTypes,1,sizeof(ValueParams<float32>));
        ctx->registerOp(OpType::DivVec,AllTypes,3,sizeof(int64_t));

        ctx->registerOp(OpType::Relu,AllTypes,1,sizeof(int64_t));
        ctx->registerOp(OpType::Silu,FloatTypes,2,sizeof(int64_t));
        ctx->registerOp(OpType::Tanh,FloatTypes,2,sizeof(int64_t));
        ctx->registerOp(OpType::Sidmoid,FloatTypes,2,sizeof(int64_t));
        ctx->registerOp(OpType::Softmax,FloatTypes,2,sizeof(SoftmaxParams));


        ctx->registerOp(OpType::Random,AllTypes,1,sizeof(RandomParams));
        ctx->registerOp(OpType::Fill,AllTypes,1,sizeof(ValueParams<float32>));

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
