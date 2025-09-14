#include "backend/cuda/ops/typecast.h"

namespace ops {

template <typename T, typename R>
__global__ void cast_cuda(const T* src, R* dst, size_t size) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        // 对于 half/bfloat16 使用专门的转换函数
        if constexpr (std::is_same_v<T, __half> && std::is_same_v<R, float>) { // half --> float32
            dst[idx] = __half2float(src[idx]);
        }else if constexpr (std::is_same_v<T, float> && std::is_same_v<R, __half>) { // float32 --> half
            dst[idx] = __float2half(src[idx]);
        }else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<R, float>) { // bfloat16 --> float32
            dst[idx] = __bfloat162float(src[idx]);
        }else if constexpr (std::is_same_v<T, float> && std::is_same_v<R, __nv_bfloat16>) { // float32 --> bfloat16
            dst[idx] = __float2bfloat16(src[idx]);
        }else if constexpr (std::is_same_v<T, __half> && std::is_same_v<R, __nv_bfloat16>) { // half --> bfloat16
            dst[idx] = __float2bfloat16(__half2float(src[idx]));
        }else if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<R, __half>) { // bfloat16 --> half
            dst[idx] = __float2half(__bfloat162float(src[idx]));
        }else {
            dst[idx] = static_cast<R>(src[idx]);
        }
    }    
}

Tensor TypecastImpl<Device::CUDA>::execute(const Tensor& a, DataType dst_type) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    // 使用模板lambda处理目标类型
    auto cast_visitor = [&]<typename DstT>(DstT* ptr_B) {
        // 根据源类型分发
        switch (a.dtype()) {
            case DataType::INT8:          
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int8_t*>(a.data()), ptr_B, numel);break;
            case DataType::INT16:         
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int16_t*>(a.data()), ptr_B, numel);break;
            case DataType::INT32:         
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int32_t*>(a.data()), ptr_B, numel); break;
            case DataType::INT64:         
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const int64_t*>(a.data()), ptr_B, numel);break;
            case DataType::FLOAT16:      
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), ptr_B, numel);break;
            case DataType::BFLOAT16:      
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), ptr_B, numel);break;
            case DataType::FLOAT32:      
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float32*>(a.data()), ptr_B, numel);break;
            case DataType::FLOAT64:      
                cast_cuda<<<blocks,threads,0,ctx_impl->stream()>>>(static_cast<const float64*>(a.data()), ptr_B, numel);break;
            default: throw std::runtime_error("Unsupported source dtype");
        }
    };
    // 获取目标数据指针并分发
    Tensor result(a.shape(), dst_type, Device::CUDA); // float32
    switch (dst_type) {
        case DataType::INT8:
            cast_visitor(static_cast<int8_t*>(result.data()));break;
        case DataType::INT16:
            cast_visitor(static_cast<int16_t*>(result.data()));break;
        case DataType::INT32:
            cast_visitor(static_cast<int32_t*>(result.data()));break;
        case DataType::INT64:
            cast_visitor(static_cast<int64_t*>(result.data()));break;
        case DataType::FLOAT16:
            cast_visitor(static_cast<__half*>(result.data())); break;
        case DataType::BFLOAT16:
            cast_visitor(static_cast<__nv_bfloat16*>(result.data()));break;
        case DataType::FLOAT32:
            cast_visitor(static_cast<float*>(result.data()));break;
        case DataType::FLOAT64:
            cast_visitor(static_cast<double*>(result.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait(); // 等待CUDA流完成
    return result;
}
template struct TypecastImpl<Device::CUDA>;

}