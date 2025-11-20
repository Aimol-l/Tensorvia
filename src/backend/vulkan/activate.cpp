
#include "backend/vulkan/ops/activate.h"

namespace ops {

 void ReluImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    
}
Tensor ReluImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result(a.shape(),a.dtype(),a.device());
    return result;
}

 void SiluImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    
}
Tensor SiluImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}

void TanhImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor TanhImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void SigmoidImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor SigmoidImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}

Tensor SoftmaxImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    int dims = a.shape().size();
    if (axis < 0) axis += dims;  // 支持负轴索引
    // 计算沿指定轴的维度信息
    size_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) {
        outer_dim *= a.shape(i);
    }
    size_t axis_dim = a.shape(axis);
    size_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) {
        inner_dim *= a.shape()[i];
    }
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }
    Tensor result(a.shape(),res_type,a.device());
   
    return result;
}

template struct ReluImpl<Device::VULKAN>;
template struct SiluImpl<Device::VULKAN>;
template struct SigmoidImpl<Device::VULKAN>;
template struct TanhImpl<Device::VULKAN>;
template struct SoftmaxImpl<Device::VULKAN>;

}