#include "backend/vulkan/ops/mul.h"


namespace ops {
// [w,h] @ [h,w] --> [w,w]
// [b,w,h] @ [b,h,w] --> [b,w,w]
Tensor MulImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b){
    int batch =     a.shape().size() == 3?a.shape(0):1;
    int rows =      a.shape().size() == 3?a.shape(1):a.shape(0);
    int common =    a.shape().size() == 3?a.shape(2):a.shape(1);
    int cols =      a.shape().size() == 3?b.shape(2):b.shape(1);
    std::vector<int64_t> newshape;
    if(a.shape().size() == 3){
        newshape = {batch,rows,cols};
    }else{
        newshape = {rows,cols};
    }
    DataType res_type = compute_type(a.dtype(),b.dtype());
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result(newshape,res_type,Device::VULKAN);
    return result;
}

template struct MulImpl<Device::VULKAN>;
}