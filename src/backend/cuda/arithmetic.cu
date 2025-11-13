#include <cmath>
#include "ops.h"
#include "backend/cuda/ops/arithmetic.h"

namespace ops {


template <typename T>
__global__ void add_cuda(const T* a_ptr,T* out_ptr,float b, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  
        out_ptr[i] = a_ptr[i] + T(b);
}
void AddImpl<Device::CUDA>::execute(Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            add_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            add_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),b,numel);
        }else{
            add_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor AddImpl<Device::CUDA>::execute(const Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());

    Tensor res(a.shape(),a.dtype(),a.device());

    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            add_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            add_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),b,numel);
        }else{
            add_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(res.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
    return res;
}
template <typename T,typename R = T>
__global__ void add_cuda(const T* a_ptr,const R* b_ptr,T* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  
        out_ptr[i] = a_ptr[i] + b_ptr[i];
}
Tensor AddImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 计算公共类别
    DataType res_type = compute_type(a.dtype(),b.dtype());
    Tensor res(a.shape(),res_type,a.device());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    switch (res_type) {
        case DataType::INT8:            
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(A.data()),static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(res.data()),numel);break;
        case DataType::INT16:           
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(A.data()),static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(res.data()), numel);break;
        case DataType::INT32:           
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(A.data()),static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(res.data()), numel);break;
        case DataType::INT64:           
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(A.data()),static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(A.data()),static_cast<const __half*>(B.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(A.data()),static_cast<const __nv_bfloat16*>(B.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(A.data()),static_cast<const float32*>(B.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(A.data()),static_cast<const float64*>(B.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for add");
    }
    ctx_impl->wait(); 
    return res;
}
void AddImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b,Tensor& dst){
    DataType res_type = compute_type(a.dtype(),b.dtype());
    if(dst.dtype() != res_type){
        throw std::runtime_error("dst dtype error!");
    }
    constexpr size_t threads = 256;
    size_t blocks = (a.numel() + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 快速路径：相同类型且无需转换
    if(a.dtype() == b.dtype()){
        dispatch_dtype(a.dtype(), [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            if constexpr(std::is_same_v<T,float16>){
                add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),static_cast<const __half*>(b.data()), static_cast<__half*>(dst.data()), a.numel());
            }else if constexpr(std::is_same_v<T,bfloat16>){
                add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),static_cast<const __nv_bfloat16*>(b.data()), static_cast<__nv_bfloat16*>(dst.data()), a.numel());
            }else{
                add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const T*>(a.data()),static_cast<const T*>(b.data()), static_cast<T*>(dst.data()), a.numel());
            }
        });
        ctx_impl->wait(); 
        return;
    }
    // 慢速路径：类型不同，需要 Typecast
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        if constexpr(std::is_same_v<T,float16>){
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(A.data()),static_cast<const __half*>(B.data()), static_cast<__half*>(dst.data()), a.numel());
        }else if constexpr(std::is_same_v<T,bfloat16>){
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(A.data()),static_cast<const __nv_bfloat16*>(B.data()), static_cast<__nv_bfloat16*>(dst.data()), a.numel());
        }else{
            add_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const T*>(A.data()),static_cast<const T*>(B.data()), static_cast<T*>(dst.data()), a.numel());
        }
    });
    ctx_impl->wait(); 
}

template <typename T>
__global__ void sub_cuda(const T* a_ptr,T* out_ptr,float b, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  
        out_ptr[i] = a_ptr[i] - T(b);
}
void SubImpl<Device::CUDA>::execute(Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            sub_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            sub_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),b,numel);
        }else{
            sub_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
}
template <typename T,typename R = T>
__global__ void sub_cuda(const T* a_ptr,const R* b_ptr,T* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  
        out_ptr[i] = a_ptr[i] - b_ptr[i];
}
Tensor SubImpl<Device::CUDA>::execute(const Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    Tensor res(a.shape(),a.dtype(),a.device());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            sub_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            sub_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),b,numel);
        }else{
            sub_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(res.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
    return res;
}
Tensor SubImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 计算公共类别
    DataType res_type = compute_type(a.dtype(),b.dtype());
    Tensor res(a.shape(),res_type,a.device());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    switch (res_type) {
        case DataType::INT8:            
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(A.data()),static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(res.data()),numel);break;
        case DataType::INT16:           
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(A.data()),static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(res.data()), numel);break;
        case DataType::INT32:           
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(A.data()),static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(res.data()), numel);break;
        case DataType::INT64:           
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(A.data()),static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(A.data()),static_cast<const __half*>(B.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(A.data()),static_cast<const __nv_bfloat16*>(B.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(A.data()),static_cast<const float32*>(B.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            sub_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(A.data()),static_cast<const float64*>(B.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for add");
    }
    ctx_impl->wait(); 
    return res;
}


template <typename T>
__global__ void dot_cuda(const T* a_ptr,T* out_ptr,float b, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)
        out_ptr[i] = a_ptr[i] * T(b);
}
void DotImpl<Device::CUDA>::execute(Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            dot_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            dot_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),b,numel);
        }else{
            dot_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor DotImpl<Device::CUDA>::execute(const Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    Tensor res(a.shape(),a.dtype(),a.device());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            dot_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            dot_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),b,numel);
        }else{
            dot_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(res.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
    return res;
}
template <typename T>
__global__ void dot_cuda(const T* a_ptr,const T* b_ptr,T* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  
        out_ptr[i] = a_ptr[i] * b_ptr[i];
}
Tensor DotImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 计算公共类别
    DataType res_type = compute_type(a.dtype(),b.dtype());
    Tensor res(a.shape(),res_type,a.device());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    switch (res_type) {
        case DataType::INT8:            
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(A.data()),static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(res.data()),numel);break;
        case DataType::INT16:           
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(A.data()),static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(res.data()), numel);break;
        case DataType::INT32:           
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(A.data()),static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(res.data()), numel);break;
        case DataType::INT64:           
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(A.data()),static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(A.data()),static_cast<const __half*>(B.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(A.data()),static_cast<const __nv_bfloat16*>(B.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(A.data()),static_cast<const float32*>(B.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            dot_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(A.data()),static_cast<const float64*>(B.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for add");
    }
    ctx_impl->wait(); 
    return res;
}



template <typename T>
__global__ void div_cuda(const T* a_ptr,T* out_ptr,float b, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  out_ptr[i] = a_ptr[i] / T(b);
}
template <typename T,typename R = T>
__global__ void div_cuda(const T* a_ptr,const T* b_ptr,R* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size)  out_ptr[i] = R(a_ptr[i]) / R(b_ptr[i]);
}
void DivImpl<Device::CUDA>::execute(Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            div_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            div_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),b,numel);
        }else{
            div_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor DivImpl<Device::CUDA>::execute(const Tensor& a,float b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    Tensor res(a.shape(),a.dtype(),a.device());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            div_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),b,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            div_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),b,numel);
        }else{
            div_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(res.data()),b,numel);
        }
    }, A);
    ctx_impl->wait();
    return res;
}
Tensor DivImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 计算公共类别
    DataType res_type = compute_type(a.dtype(),b.dtype());
    Tensor res(a.shape(),res_type,a.device());
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    switch (res_type) {
        case DataType::INT8:            
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(A.data()),static_cast<const int8_t*>(B.data()), static_cast<int8_t*>(res.data()),numel);break;
        case DataType::INT16:           
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(A.data()),static_cast<const int16_t*>(B.data()), static_cast<int16_t*>(res.data()), numel);break;
        case DataType::INT32:           
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(A.data()),static_cast<const int32_t*>(B.data()), static_cast<int32_t*>(res.data()), numel);break;
        case DataType::INT64:           
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(A.data()),static_cast<const int64_t*>(B.data()), static_cast<int64_t*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(A.data()),static_cast<const __half*>(B.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(A.data()),static_cast<const __nv_bfloat16*>(B.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(A.data()),static_cast<const float32*>(B.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            div_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(A.data()),static_cast<const float64*>(B.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for div");
    }
    ctx_impl->wait(); 
    return res;
}

//********************************************************************
template <typename T,typename R = T>
__global__ void sin_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(sinf(__half2float(src_ptr[tid])));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(sinf(__bfloat162float(src_ptr[tid])));
        }else{
            dst_ptr[tid] = R(sinf(float(src_ptr[tid])));
        }
    }
}
void SinImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            sin_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            sin_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            sin_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}

Tensor SinImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),numel);break;
        case DataType::INT16:           
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT32:           
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT64:           
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            sin_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for sin");
    }
    ctx_impl->wait(); 
    return res;
}


template <typename T,typename R = T>
__global__ void cos_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(cosf(__half2float(src_ptr[tid])));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(cosf(__bfloat162float(src_ptr[tid])));
        }else{
            dst_ptr[tid] = R(cosf(float(src_ptr[tid])));
        }
    }
}
void CosImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            cos_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            cos_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            cos_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}

Tensor CosImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),numel);break;
        case DataType::INT16:           
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT32:           
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT64:           
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            cos_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for cos");
    }
    ctx_impl->wait(); 
    return res;
}


template <typename T,typename R = T>
__global__ void tan_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(tanf(__half2float(src_ptr[tid])));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(tanf(__bfloat162float(src_ptr[tid])));
        }else{
            dst_ptr[tid] = R(tanf(float(src_ptr[tid])));
        }
    }
}
void TanImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            tan_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            tan_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            tan_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor TanImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),numel);break;
        case DataType::INT16:           
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT32:           
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT64:           
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            tan_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for cos");
    }
    ctx_impl->wait(); 
    return res;
}

template <typename T,typename R = T>
__global__ void pow_cuda(const T* src_ptr, R* dst_ptr,float val, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(powf(__half2float(src_ptr[tid]),val));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(powf(__bfloat162float(src_ptr[tid]),val));
        }else{
            dst_ptr[tid] = R(powf(float(src_ptr[tid]),val));
        }
    }
}
void PowImpl<Device::CUDA>::execute(Tensor& a,float val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            pow_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),val,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            pow_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),val,numel);
        }else{
            pow_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()),val,numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor PowImpl<Device::CUDA>::execute(const Tensor& a,float val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT16:           
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT32:           
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT64:           
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::FLOAT16:         
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),val,numel);break;
        case DataType::BFLOAT16:        
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),val,numel);break;
        case DataType::FLOAT32:         
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::FLOAT64:         
            pow_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()),val,numel);break;
        default: throw std::runtime_error("Unsupported dtype for cos");
    }
    ctx_impl->wait(); 
    return res;
}

template <typename T,typename R = T>
__global__ void log_cuda(const T* src_ptr, R* dst_ptr,float val, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(powf(__half2float(src_ptr[tid]),val));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(powf(__bfloat162float(src_ptr[tid]),val));
        }else{
            dst_ptr[tid] = R(powf(float(src_ptr[tid]),val));
        }
    }
}
void LogImpl<Device::CUDA>::execute(Tensor& a,float val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            log_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),val,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            log_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),val,numel);
        }else{
            log_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()),val,numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor LogImpl<Device::CUDA>::execute(const Tensor& a,float val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT16:           
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT32:           
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::INT64:           
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::FLOAT16:         
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),val,numel);break;
        case DataType::BFLOAT16:        
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),val,numel);break;
        case DataType::FLOAT32:         
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()),val,numel);break;
        case DataType::FLOAT64:         
            log_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()),val,numel);break;
        default: throw std::runtime_error("Unsupported dtype for cos");
    }
    ctx_impl->wait(); 
    return res;
}

template <typename T,typename R = T>
__global__ void exp_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(expf(__half2float(src_ptr[tid])));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(expf(__bfloat162float(src_ptr[tid])));
        }else{
            dst_ptr[tid] = R(expf(float(src_ptr[tid])));
        }
    }
}
void ExpImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            exp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            exp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            exp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor ExpImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),numel);break;
        case DataType::INT16:           
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT32:           
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT64:           
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            exp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for exp");
    }
    ctx_impl->wait(); 
    return res;
}


template <typename T,typename R = T>
__global__ void sqrt_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        if constexpr(std::is_same_v<R,__half>) {
            dst_ptr[tid] = __half2float(sqrtf(__half2float(src_ptr[tid])));
        }else if(std::is_same_v<R,__nv_bfloat16>){
            dst_ptr[tid] = __bfloat162float(sqrtf(__bfloat162float(src_ptr[tid])));
        }else{
            dst_ptr[tid] = R(sqrtf(float(src_ptr[tid])));
        }
    }
}
void SqrtImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            sqrt_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            sqrt_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            sqrt_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor SqrtImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)   res_type = DataType::FLOAT32;
    Tensor res(a.shape(),res_type,a.device());
    switch (res_type) {
        case DataType::INT8:            
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()),numel);break;
        case DataType::INT16:           
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT32:           
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::INT64:           
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            sqrt_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for sqrt");
    }
    ctx_impl->wait(); 
    return res;
}

template <typename T>
__global__ void abs_cuda(const T* src_ptr, T* dst_ptr, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        T x = src_ptr[tid];
        // 处理不同数据类型的绝对值计算
        if constexpr (std::is_same_v<T, __half>) {
            uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
            *x_ptr &= 0x7FFF; // 清除最高位（符号位）
            dst_ptr[tid] = x;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
            *x_ptr &= 0x7FFF; // 清除最高位（符号位）
            dst_ptr[tid] = x;
        } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            dst_ptr[tid] = std::abs(x);
        } else if constexpr (std::is_floating_point_v<T>) {
            dst_ptr[tid] = std::fabs(x);
        } else {
            dst_ptr[tid] = x;
        }
    }
}

void AbsImpl<Device::CUDA>::execute(Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            abs_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            abs_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            abs_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()), numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor AbsImpl<Device::CUDA>::execute(const Tensor& a){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    Tensor res(a.shape(),a.dtype(),a.device());
    switch (a.dtype()) {
        case DataType::INT8:            
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(res.data()),numel);break;
        case DataType::INT16:           
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(res.data()), numel);break;
        case DataType::INT32:           
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(res.data()), numel);break;
        case DataType::INT64:           
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(res.data()), numel);break;
        case DataType::FLOAT16:         
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()), numel);break;
        case DataType::BFLOAT16:        
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()), numel);break;
        case DataType::FLOAT32:         
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), numel);break;
        case DataType::FLOAT64:         
            abs_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for sqrt");
    }
    ctx_impl->wait(); 
    return res;
}

template <typename T>
__global__ void clamp_cuda(const T* src_ptr, T* dst_ptr, float min_val, float max_val, size_t numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        T x = src_ptr[tid];
        // 处理不同数据类型的 clamp 操作
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            // 对于半精度类型，转换为 float 进行比较，再转回
            float x_float;
            if constexpr (std::is_same_v<T, __half>) {
                x_float = __half2float(x);
            } else {
                x_float = __bfloat162float(x);
            }
            float clamped = fminf(fmaxf(x_float, min_val), max_val);
            if constexpr (std::is_same_v<T, __half>) {
                dst_ptr[tid] = __float2half(clamped);
            } else {
                dst_ptr[tid] = __float2bfloat16(clamped);
            }
        } else if constexpr (std::is_integral_v<T>) {
            // 对于整数类型，直接比较和赋值
            // T min_t = static_cast<T>(min_val);
            // T max_t = static_cast<T>(max_val);
            dst_ptr[tid] = static_cast<T>(fminf(max_val, std::fminf(min_val, float(x))));
        } else {
            // 对于浮点数类型，使用 fminf 和 fmaxf
            dst_ptr[tid] = fminf(fmaxf(x, min_val), max_val);
        }
    }
}
void ClampImpl<Device::CUDA>::execute(Tensor& a,float min_val,float max_val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) { 
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        if constexpr(std::is_same_v<AType,float16>){
            clamp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()),min_val, max_val,numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            clamp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()),min_val, max_val, numel);
        }else{
            clamp_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(a.data()),min_val, max_val, numel);
        }
    }, A);
    ctx_impl->wait();
}
Tensor ClampImpl<Device::CUDA>::execute(const Tensor& a,float min_val,float max_val){
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    Tensor res(a.shape(),a.dtype(),a.device());
    switch (a.dtype()) {
        case DataType::INT8:            
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(res.data()),min_val, max_val,numel);break;
        case DataType::INT16:           
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(res.data()),min_val, max_val, numel);break;
        case DataType::INT32:           
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(res.data()),min_val, max_val, numel);break;
        case DataType::INT64:           
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(res.data()),min_val, max_val, numel);break;
        case DataType::FLOAT16:         
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(res.data()),min_val, max_val, numel);break;
        case DataType::BFLOAT16:        
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(res.data()),min_val, max_val, numel);break;
        case DataType::FLOAT32:         
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()),min_val, max_val, numel);break;
        case DataType::FLOAT64:         
            clamp_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()),min_val, max_val, numel);break;
        default: throw std::runtime_error("Unsupported dtype for clamp");
    }
    ctx_impl->wait(); 
    return res;
}


template struct AddImpl<Device::CUDA>;
template struct SubImpl<Device::CUDA>;
template struct DotImpl<Device::CUDA>;
template struct DivImpl<Device::CUDA>;
template struct SinImpl<Device::CUDA>;
template struct CosImpl<Device::CUDA>;
template struct TanImpl<Device::CUDA>;
template struct PowImpl<Device::CUDA>;
template struct LogImpl<Device::CUDA>;
template struct ExpImpl<Device::CUDA>;
template struct SqrtImpl<Device::CUDA>;
template struct AbsImpl<Device::CUDA>;
template struct ClampImpl<Device::CUDA>;

}