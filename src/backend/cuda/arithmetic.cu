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

    const Tensor& A = a.dtype() == res_type ? a : ops::typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::typecast(b,res_type);

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
    const Tensor& A = a.dtype() == res_type ? a : ops::typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::typecast(b,res_type);
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
    const Tensor& A = a.dtype() == res_type ? a : ops::typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::typecast(b,res_type);
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
    const Tensor& A = a.dtype() == res_type ? a : ops::typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::typecast(b,res_type);
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




// //********************************************************************
// template <typename T,typename R = T>
// __global__ void sin_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(sinf(float(src_ptr[tid])));
//         }else{
//             dst_ptr[tid] = R(sinf(R(src_ptr[tid])));
//         }
//     }
// }
// template <typename T,typename R = T>
// __global__ void cos_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(cosf(float(src_ptr[tid])));
//         }else{
//             dst_ptr[tid] = R(cosf(R(src_ptr[tid])));
//         }
//     }
// }

// template <typename T,typename R = T>
// __global__ void tan_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(tanf(float(src_ptr[tid])));
//         }else{
//             dst_ptr[tid] = R(tanf(R(src_ptr[tid])));
//         }
//     }
// }
// template <typename T,typename R = T>
// __global__ void pow_cuda(const T* src_ptr, R* dst_ptr,float val, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(powf(float(src_ptr[tid]),float(val)));
//         }else{
//             dst_ptr[tid] = R(powf(R(src_ptr[tid]),R(val)));
//         }
//     }
// }
// template <typename T,typename R = T>
// __global__ void log_cuda(const T* src_ptr, R* dst_ptr, float val,size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(logf(float(src_ptr[tid])) / logf(val));
//         }else{
//             dst_ptr[tid] = R(logf(R(src_ptr[tid]) / logf(R(val))));
//         }
//     }
// }
// template <typename T,typename R = T>
// __global__ void exp_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(expf(float(src_ptr[tid]),float(val)));
//         }else{
//             dst_ptr[tid] = R(expf(R(src_ptr[tid]),R(val)));
//         }
//     }
// }
// template <typename T,typename R = T>
// __global__ void sqrt_cuda(const T* src_ptr, R* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         if constexpr(std::is_same_v<R,__half> || std::is_same_v<R,__nv_bfloat16>){
//             dst_ptr[tid] = R(sqrtf(float(src_ptr[tid]),float(val)));
//         }else{
//             dst_ptr[tid] = R(sqrtf(R(src_ptr[tid]),R(val)));
//         }
//     }
// }
// template <typename T>
// __global__ void abs_cuda(const T* src_ptr, T* dst_ptr, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         dst_ptr[tid] = abs(x);
//     }
// }
// template <typename T>
// __global__ void clamp_cuda(const T* src_ptr, T* dst_ptr,float min_val,float max_val, size_t numel) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < numel) {
//         dst_ptr[tid] = src_ptr[i] > max_val ? max_val : (src_ptr[i] < min_val ? min_val : src_ptr[i]);
//     }
// }
// *****************************************





void SinImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor SinImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void CosImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor CosImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void TanImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor TanImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void PowImpl<Device::CUDA>::execute(Tensor& a,float val){

}
Tensor PowImpl<Device::CUDA>::execute(const Tensor& a,float val){
    return a.clone();
}
void LogImpl<Device::CUDA>::execute(Tensor& a,float val){

}
Tensor LogImpl<Device::CUDA>::execute(const Tensor& a,float val){
    return a.clone();
}
void ExpImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor ExpImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void SqrtImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor SqrtImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void AbsImpl<Device::CUDA>::execute(Tensor& a){

}
Tensor AbsImpl<Device::CUDA>::execute(const Tensor& a){
    return a.clone();
}
void ClampImpl<Device::CUDA>::execute(Tensor& a,float min_val,float max_val){

}
Tensor ClampImpl<Device::CUDA>::execute(const Tensor& a,float min_val,float max_val){
    return a.clone();
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