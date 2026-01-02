
#include "backend/vulkan/ops/activate.h"
using namespace via;

namespace ops {

 void ReluImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel(); 
    ctx_impl->submitCompute(
        OpType::Relu, 
        a.dtype(),
        {src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
}
Tensor ReluImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.clone();
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Relu, 
        result.dtype(),
        {src_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    
    return result;
}

 void SiluImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Silu, 
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    
}
Tensor SiluImpl<Device::VULKAN>::execute(const Tensor& a){
    
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();

    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());

    int64_t numel = result.numel(); 

    ctx_impl->submitCompute(
        OpType::Silu, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    
    return result;
}

void TanhImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Tanh, 
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
}
Tensor TanhImpl<Device::VULKAN>::execute(const Tensor& a){
   Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();

    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());

    int64_t numel = result.numel(); 

    ctx_impl->submitCompute(
        OpType::Tanh, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    
    return result;
}

void SigmoidImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Sidmoid, 
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
}
Tensor SigmoidImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();

    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());

    int64_t numel = result.numel(); 

    ctx_impl->submitCompute(
        OpType::Sidmoid, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    
    return result;
}

// shader限制，只能
Tensor SoftmaxImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    int dims = a.shape().size();
    if (axis < 0) axis += dims;  // 支持负轴索引
    // 计算沿指定轴的维度信息
    int32_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) {
        outer_dim *= a.shape(i);
    }
    int32_t axis_dim = a.shape(axis);
    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) {
        inner_dim *= a.shape()[i];
    }
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }

    // float16 --> float16
    // float32 --> float32
    // float64 --> float64
    // bfloat16 --> bfloat16
    // int8,int16,int32 --> float32
    Tensor result(a.shape(),res_type,a.device());
   
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());


    SoftmaxParams params{
        .axis_dim = axis_dim,
        .outer_dim = outer_dim,
        .inner_dim = inner_dim
    };

    ctx_impl->submitCompute(
        OpType::Softmax, 
        result.dtype(),
        {src_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );

    return result;
}

template struct ReluImpl<Device::VULKAN>;
template struct SiluImpl<Device::VULKAN>;
template struct SigmoidImpl<Device::VULKAN>;
template struct TanhImpl<Device::VULKAN>;
template struct SoftmaxImpl<Device::VULKAN>;

}