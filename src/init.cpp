#include "core/factory.h"
#include "backend/cpu/cpu_tensor.h"


#ifdef BACKEND_CPU
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,std::vector<int64_t> shape, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,shape, dtype);
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
        register_tensor_impl(Device::CUDA, [&](void* ptr,std::vector<int64_t> shape, DataType dtype) {
            return std::make_shared<CUDATensor>(ptr,shape, dtype,ctx);
        });
    }
}cuda_registrar;
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,std::vector<int64_t> shape, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,shape, dtype);
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
        register_tensor_impl(Device::SYCL, [&](void* ptr,std::vector<int64_t> shape, DataType dtype) {
            return std::make_shared<SYCLTensor>(ptr,shape, dtype,ctx);
        });
    }
} sycl_registrar;
struct CPURegistrar {
    CPURegistrar() {
        register_tensor_impl(Device::CPU, [](void* ptr,std::vector<int64_t> shape, DataType dtype) {
            return std::make_shared<CPUTensor>(ptr,shape, dtype);
        });
    }
} cpu_registrar;
#endif
