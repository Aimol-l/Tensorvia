#include "backend/vulkan/ops/arithmetic.h"

namespace ops {

void AddImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
// uninplace
Tensor AddImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Add(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::VULKAN);
    return result;
}
Tensor AddImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Add(a, t);
}

void SubImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
// uninplace
Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Add(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    Tensor result(a.shape(), res_type, Device::VULKAN);

   
    return result;
}
Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Sub(a, t);
}
void DotImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    
}
Tensor DotImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Add(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    Tensor result(a.shape(), res_type, Device::VULKAN);

   
    return result;
}
Tensor DotImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Dot(a, t);
}
void DivImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

}
    // uninplace
Tensor DivImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Add(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    Tensor result(a.shape(), res_type, Device::VULKAN);

    return result;
}
    Tensor DivImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Div(a, t);
}


void SinImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
   
}
Tensor SinImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    Tensor result;
    
    return result;
}
void CosImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
   
}
Tensor CosImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}

void TanImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor TanImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void ExpImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor ExpImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void SqrtImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor SqrtImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void PowImpl<Device::VULKAN>::execute(Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor PowImpl<Device::VULKAN>::execute(const Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void LogImpl<Device::VULKAN>::execute(Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor LogImpl<Device::VULKAN>::execute(const Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result;
    return result;
}
void ClampImpl<Device::VULKAN>::execute(Tensor& a,float min,float max){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor ClampImpl<Device::VULKAN>::execute(const Tensor& a,float min,float max){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result(a.shape(),a.dtype(),a.device());
    return result;
}

void AbsImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor AbsImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    Tensor result(a.shape(),a.dtype(),a.device());
    return result;
}

 template struct AddImpl<Device::VULKAN>;
 template struct SubImpl<Device::VULKAN>;
 template struct DotImpl<Device::VULKAN>;
 template struct DivImpl<Device::VULKAN>;
 template struct SinImpl<Device::VULKAN>;
 template struct CosImpl<Device::VULKAN>;
 template struct TanImpl<Device::VULKAN>;
 template struct ExpImpl<Device::VULKAN>;
 template struct SqrtImpl<Device::VULKAN>;
 template struct PowImpl<Device::VULKAN>;
 template struct LogImpl<Device::VULKAN>;
 template struct ClampImpl<Device::VULKAN>;
 template struct AbsImpl<Device::VULKAN>;

}