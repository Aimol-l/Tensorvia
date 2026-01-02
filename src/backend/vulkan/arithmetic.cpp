#include "ops.h"
#include "backend/vulkan/ops/arithmetic.h"
using namespace via;

namespace ops {

void AddImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    ValueParams<float> params{
        .value = b,
        .numel = static_cast<int64_t>(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Add, 
        a.dtype(),
        {src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
}
void SubImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    ValueParams<float> params{
        .value = b,
        .numel = static_cast<int64_t>(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Sub, 
        a.dtype(),
        {src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
}
void DotImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    ValueParams<float> params{
        .value = b,
        .numel = static_cast<int64_t>(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Dot, 
        a.dtype(),
        {src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
}
void DivImpl<Device::VULKAN>::execute(Tensor& a,float b){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

}

Tensor AddImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = a.clone();
    ops::Add(t, b);
    return t;
}
Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = a.clone();
    ops::Sub(t, b);
    return t;
}
Tensor DotImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Dot(a, t);
}
Tensor DivImpl<Device::VULKAN>::execute(const Tensor& a, float b){
    Tensor t = ops::Fill(a.shape(),a.dtype(),b);
    return ops::Div(a, t);
}

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

    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::VULKAN);
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    
    int64_t num = a.numel();

    ctx_impl->submitCompute(
        OpType::AddVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );

    return result;
}
Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
   // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Sub(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::VULKAN);
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    
    int64_t num = a.numel();

    ctx_impl->submitCompute(
        OpType::SubVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );

    return result;
}
Tensor DotImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    // 避免自加修改：a + a 返回新 tensor
    if (&a == &b) ops::Dot(a.clone(), b.clone());
    // 计算公共类别
    DataType res_type = std::max(a.dtype(),b.dtype()); // 全是int 或 全是 float 
    if(a.dtype() <= DataType::INT64 && b.dtype() > DataType::INT64){
        res_type = std::max(b.dtype(),DataType::FLOAT32);
    }else if(a.dtype() > DataType::INT64 && b.dtype() <= DataType::INT64){
        res_type = std::max(a.dtype(),DataType::FLOAT32);
    }
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);

    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    Tensor result(a.shape(), res_type, Device::VULKAN);
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    
    int64_t num = a.numel();

    ctx_impl->submitCompute(
        OpType::DotVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );

    return result;
}
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


void AddImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b,Tensor& dst){
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
    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    int64_t num = a.numel();
    ctx_impl->submitCompute(
        OpType::AddVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );
}
void SubImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b,Tensor& dst){
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
    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    int64_t num = a.numel();
    ctx_impl->submitCompute(
        OpType::SubVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );
}
void DotImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b,Tensor& dst){
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
    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    int64_t num = a.numel();
    ctx_impl->submitCompute(
        OpType::DotVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );
}
void DivImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b,Tensor& dst){
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
    // 到这里能保证A,B类型一致
    size_t size = a.numel();
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    int64_t num = a.numel();
    ctx_impl->submitCompute(
        OpType::DivVec, 
        A.dtype(),
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        (A.numel() + 255) / 256, 1, 1,
        &num, 
        sizeof(num)
    );
}


void AbsImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::Abs,
        a.dtype(),
        {src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
}
void SinImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Sin,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(numel)
    );
}
void CosImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Cos,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(numel)
    );
}
void TanImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Tan,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(numel)
    );
}
void ExpImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Exp,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(numel)
    );
}
void SqrtImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Sqrt,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(numel)
    );
}
void PowImpl<Device::VULKAN>::execute(Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    ValueParams<float> params{
        .value = val,
        .numel = int64_t(a.numel())
    };

    ctx_impl->submitCompute(
        OpType::Pow,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );

}
void LogImpl<Device::VULKAN>::execute(Tensor& a,float val){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    ValueParams<float> params{
        .value = val,
        .numel = int64_t(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Log,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );
}
void ClampImpl<Device::VULKAN>::execute(Tensor& a,float min,float max){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    ClampParams params{
        .min = min,
        .max = max,
        .numel = int64_t(a.numel())
    };

    ctx_impl->submitCompute(
        OpType::Clamp,
        a.dtype(),
        {src_impl->buffer(),src_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );

}


Tensor AbsImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.clone();
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());

    int64_t numel = result.numel(); 
    
    ctx_impl->submitCompute(
        OpType::Abs, 
        result.dtype(),
        {src_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor SinImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Sin, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor CosImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Cos, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor TanImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Tan, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor ExpImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Exp, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor SqrtImpl<Device::VULKAN>::execute(const Tensor& a){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    int64_t numel = result.numel(); 
    ctx_impl->submitCompute(
        OpType::Sqrt, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return result;
}
Tensor PowImpl<Device::VULKAN>::execute(const Tensor& a,float val){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    ValueParams<float> params{
        .value = val,
        .numel = int64_t(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Pow, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );
    return result;
}
Tensor LogImpl<Device::VULKAN>::execute(const Tensor& a,float val){
    Tensor result = a.dtype() < DataType::BFLOAT16 ? a.clone().to_type_(DataType::FLOAT32):a.clone();
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    ValueParams<float> params{
        .value = val,
        .numel = int64_t(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Log, 
        result.dtype(),
        {dst_impl->buffer(),dst_impl->buffer()},
        (result.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );
    return result;
}
Tensor ClampImpl<Device::VULKAN>::execute(const Tensor& a,float min,float max){
    Tensor result(a.shape(),a.dtype(),a.device());
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl =  std::dynamic_pointer_cast<VKTensor>(result.get_impl());

    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    ClampParams params{
        .min = min,
        .max = max,
        .numel = int64_t(a.numel())
    };
    ctx_impl->submitCompute(
        OpType::Clamp,
        a.dtype(),
        {src_impl->buffer(),dst_impl->buffer()},
        (a.numel() + 255) / 256, 1, 1,
        &params,
        sizeof(params)
    );

    return result;
}


template struct AddImpl<Device::VULKAN>;
template struct SubImpl<Device::VULKAN>;
template struct DotImpl<Device::VULKAN>;
template struct DivImpl<Device::VULKAN>;
template struct SinImpl<Device::VULKAN>;
template struct CosImpl<Device::VULKAN>;
template struct TanImpl<Device::VULKAN>;
template struct AbsImpl<Device::VULKAN>;
template struct ExpImpl<Device::VULKAN>;
template struct PowImpl<Device::VULKAN>;
template struct LogImpl<Device::VULKAN>;
template struct SqrtImpl<Device::VULKAN>;
template struct ClampImpl<Device::VULKAN>;
}