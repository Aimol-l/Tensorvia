#include <cmath>
#include "backend/cuda/ops/activate.h"

namespace ops {

template <typename T>
__global__ void relu_cuda(const T* src_ptr, T* dst_ptr, size_t size) {
    const size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        T x = src_ptr[glob_id];
        dst_ptr[glob_id] = x > T(0) ? x : T(0);
    }
}
template <typename T,typename R = T>
__global__ void silu_cuda(const T* src_ptr, R* dst_ptr, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        R x = src_ptr[glob_id];
        dst_ptr[glob_id] = x / R(1 + expf(-x));
    }
}
template <typename T,typename R = T>
__global__ void tanh_cuda(const T* src_ptr, R* dst_ptr, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        R x = src_ptr[glob_id];
        dst_ptr[glob_id] = tanhf(x);
    }
}
template <typename T,typename R = T>
__global__ void sigmoid_cuda(const T* src_ptr, R* dst_ptr, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        R x = src_ptr[glob_id];
        dst_ptr[glob_id] = R(1 / (1 + expf(-x)));
    }
}
template <typename T, typename R = float>
__global__ void softmax_cuda(const T* src_ptr, R* dst_ptr, size_t outer_dim, size_t axis_dim, size_t inner_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer_dim * inner_dim;
    if (idx >= total) return;
    // 计算索引
    size_t outer_idx = idx / inner_dim;
    size_t inner_idx = idx % inner_dim;
    const size_t base_offset = outer_idx * axis_dim * inner_dim + inner_idx;
    // 根据R的类型选择计算精度
    if constexpr (std::is_same_v<R, double>) {
        // 双精度计算
        double max_val = -INFINITY;
        for (size_t i = 0; i < axis_dim; ++i) {
            double val = static_cast<double>(src_ptr[base_offset + i * inner_dim]);
            max_val = fmax(val, max_val);
        }
        double sum = 0.0;
        for (size_t i = 0; i < axis_dim; ++i) {
            double val = static_cast<double>(src_ptr[base_offset + i * inner_dim]);
            sum += exp(val - max_val);
        }
        for (size_t i = 0; i < axis_dim; ++i) {
            double val = static_cast<double>(src_ptr[base_offset + i * inner_dim]);
            double result = exp(val - max_val) / sum;
            dst_ptr[base_offset + i * inner_dim] = static_cast<R>(result);
        }
    } else {
        // 单精度计算
        float max_val = -INFINITY;
        for (size_t i = 0; i < axis_dim; ++i) {
            float val = static_cast<float>(src_ptr[base_offset + i * inner_dim]);
            max_val = fmaxf(val, max_val);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < axis_dim; ++i) {
            float val = static_cast<float>(src_ptr[base_offset + i * inner_dim]);
            sum += expf(val - max_val);
        }
        for (size_t i = 0; i < axis_dim; ++i) {
            float val = static_cast<float>(src_ptr[base_offset + i * inner_dim]);
            float result = expf(val - max_val) / sum;
            dst_ptr[base_offset + i * inner_dim] = static_cast<R>(result);
        }
    }
}
//====================================================================
void ReluImpl<Device::CUDA>::execute(Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            relu_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            relu_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            relu_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),numel);
        }
    }, A);
    ctx_impl->wait();
}
void SiluImpl<Device::CUDA>::execute(Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            silu_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            silu_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            silu_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),numel);
        }
    }, A);
    ctx_impl->wait();
}
void TanhImpl<Device::CUDA>::execute(Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            tanh_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            tanh_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            tanh_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),numel);
        }
    }, A);
    ctx_impl->wait();
}
void SigmoidImpl<Device::CUDA>::execute(Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            sigmoid_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(a.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            sigmoid_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(a.data()), numel);
        }else{
            sigmoid_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),static_cast<AType*>(a.data()),numel);
        }
    }, A);
    ctx_impl->wait();
}
//====================================================================
Tensor ReluImpl<Device::CUDA>::execute(const Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    Tensor b(a.shape(), a.dtype(), a.device());
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            relu_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(b.data()), numel);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            relu_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(b.data()), numel);
        }else{
            relu_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(b.data()), numel);
        }   
    }, A, B);
    ctx_impl->wait();
    return b;
}
Tensor SiluImpl<Device::CUDA>::execute(const Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }
    Tensor b(a.shape(),res_type,a.device());
    switch (a.dtype()) {
        case DataType::INT8:            
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT16:           
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT32:           
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT64:           
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        case DataType::FLOAT16:         
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),static_cast<__half*>(b.data()), numel);break;
        case DataType::BFLOAT16:        
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),static_cast<__nv_bfloat16*>(b.data()), numel);break;
        case DataType::FLOAT32:         
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::FLOAT64:         
            silu_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for silu");
    }
    ctx_impl->wait();   
    return b;
}
Tensor TanhImpl<Device::CUDA>::execute(const Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }
    Tensor b(a.shape(),res_type,a.device());
    switch (a.dtype()) {
        case DataType::INT8:            
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT16:           
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT32:           
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT64:           
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        case DataType::FLOAT16:         
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),static_cast<__half*>(b.data()), numel);break;
        case DataType::BFLOAT16:        
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),static_cast<__nv_bfloat16*>(b.data()), numel);break;
        case DataType::FLOAT32:         
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::FLOAT64:         
            tanh_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for silu");
    }
    ctx_impl->wait();   
    return b;
}
Tensor SigmoidImpl<Device::CUDA>::execute(const Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }
    Tensor b(a.shape(),res_type,a.device());
    switch (a.dtype()) {
        case DataType::INT8:            
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT16:           
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT32:           
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::INT64:           
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        case DataType::FLOAT16:         
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),static_cast<__half*>(b.data()), numel);break;
        case DataType::BFLOAT16:        
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),static_cast<__nv_bfloat16*>(b.data()), numel);break;
        case DataType::FLOAT32:         
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()),static_cast<float32*>(b.data()), numel);break;
        case DataType::FLOAT64:         
            sigmoid_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()),static_cast<float64*>(b.data()), numel);break;
        default: throw std::runtime_error("Unsupported dtype for silu");
    }
    ctx_impl->wait();   
    return b;
}
Tensor SoftmaxImpl<Device::CUDA>::execute(const Tensor& a,int axis) {
    int dims = a.shape().size();
    if (axis < 0) axis += dims;  // 支持负轴索引
    // 计算沿指定轴的维度信息
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    size_t axis_dim = a.shape(axis);
    for (int i = 0; i < axis; ++i) {
        outer_dim *= a.shape(i);
    }
    for (int i = axis + 1; i < dims; ++i) {
        inner_dim *= a.shape(i);
    }
    constexpr size_t threads = 256;
    size_t total = outer_dim * inner_dim;
    size_t blocks = (total + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    
    DataType res_type = a.dtype();
    if(res_type <= DataType::INT32){
        res_type = DataType::FLOAT32;
    }else if(res_type == DataType::INT64||res_type== DataType::FLOAT64){
        res_type = DataType::FLOAT64;
    }
    Tensor result(a.shape(),res_type,a.device());
    switch (a.dtype()) {
        case DataType::INT8:        
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()),static_cast<float32*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::INT16:       
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()),static_cast<float32*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::INT32:       
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()),static_cast<float32*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::INT64:       
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()),static_cast<float64*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::FLOAT16:     
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),static_cast<__half*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::BFLOAT16:    
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),static_cast<__nv_bfloat16*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::FLOAT32:     
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()),static_cast<float32*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        case DataType::FLOAT64:     
            softmax_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()),static_cast<float64*>(result.data()),outer_dim,axis_dim,inner_dim); break;
        default: throw std::runtime_error("softmax: unsupported data type");
    }
    ctx_impl->wait();
    return result;
}
//====================================================================
template struct ReluImpl<Device::CUDA>;
template struct SiluImpl<Device::CUDA>;
template struct TanhImpl<Device::CUDA>;
template struct SigmoidImpl<Device::CUDA>;
template struct SoftmaxImpl<Device::CUDA>;
}
