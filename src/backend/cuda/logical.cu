#include "ops.h"
#include "backend/cuda/ops/logical.h"

namespace ops {

template <typename T,typename R>
__global__ void equal_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] == b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = (abs(val_a - val_b) <= max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor EqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) { 
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                equal_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                equal_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                equal_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                equal_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                equal_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                equal_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                equal_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                equal_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T,typename R>
__global__ void not_equal_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] != b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = (abs(val_a - val_b) > max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor NotEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                not_equal_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                not_equal_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                not_equal_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                not_equal_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                not_equal_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                not_equal_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                not_equal_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                not_equal_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T,typename R>
__global__ void greater_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] > b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = ((val_a - val_b) > max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor GreaterImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                greater_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                greater_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                greater_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                greater_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                greater_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                greater_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                greater_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                greater_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T,typename R>
__global__ void less_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] < b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = ((val_a - val_b) < max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor LessImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                less_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                less_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                less_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                less_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                less_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                less_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                less_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                less_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T,typename R>
__global__ void greater_equal_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] >= b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = ((val_a - val_b) >= max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor GreaterEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                greater_equal_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                greater_equal_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                greater_equal_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                greater_equal_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                greater_equal_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                greater_equal_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                greater_equal_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                greater_equal_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T,typename R>
__global__ void less_equal_cuda(const T* a_ptr, const R* b_ptr, int8_t* out_ptr, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    constexpr PromotedType abs_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    constexpr PromotedType rel_tol = [] {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
        else if constexpr (std::is_same_v<T, __half> || std::is_same_v<R, __half>) return static_cast<PromotedType>(1e-3);
        else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<R, __nv_bfloat16>) return static_cast<PromotedType>(1e-2);
        return PromotedType{0};
    }();
    if (i < size) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            out_ptr[i] = (a_ptr[i] <= b_ptr[i]) ? 1 : 0;
        } else {
            // 至少存在一个浮点
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            out_ptr[i] = ((val_a - val_b) <= max(rel_tol * max(abs(val_a), abs(val_b)), abs_tol)) ? 1 : 0;
        }
    }
}
Tensor LessEqualImpl<Device::CUDA>::execute(const Tensor& a,const Tensor& b) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CUDA);
    auto b_visitor = [&]<typename T>(const T* ptr_a){
        switch (b.dtype()) {
            case DataType::INT8:
                less_equal_cuda<T,int8_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int8_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT16:
                less_equal_cuda<T,int16_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int16_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT32:
                less_equal_cuda<T,int32_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int32_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::INT64:
                less_equal_cuda<T,int64_t><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const int64_t*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT16:
                less_equal_cuda<T,__half><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __half*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::BFLOAT16:
                less_equal_cuda<T,__nv_bfloat16><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const __nv_bfloat16*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT32:
                less_equal_cuda<T,float><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const float*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            case DataType::FLOAT64:
                less_equal_cuda<T,double><<<blocks,threads,0,ctx_impl->stream()>>>(ptr_a,static_cast<const double*>(b.data()),static_cast<int8_t*>(res.data()), numel);break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };
    switch (a.dtype()) {
        case DataType::INT8:
            b_visitor(static_cast<const int8_t*>(a.data()));break;
        case DataType::INT16:
            b_visitor(static_cast<const int16_t*>(a.data()));break;
        case DataType::INT32:
            b_visitor(static_cast<const int32_t*>(a.data()));break;
        case DataType::INT64:
            b_visitor(static_cast<const int64_t*>(a.data()));break;
        case DataType::FLOAT16:
            b_visitor(static_cast<const __half*>(a.data())); break;
        case DataType::BFLOAT16:
            b_visitor(static_cast<const __nv_bfloat16*>(a.data()));break;
        case DataType::FLOAT32:
            b_visitor(static_cast<const float*>(a.data()));break;
        case DataType::FLOAT64:
            b_visitor(static_cast<const double*>(a.data()));break;
        default: throw std::runtime_error("Unsupported destination dtype");
    }
    ctx_impl->wait();
    return res;
}

template <typename T>
__global__ void non_zero_cuda(const T* __restrict__ a_ptr,size_t size,unsigned long long* result) {
    // 动态共享内存，每个线程存一个局部计数
    extern __shared__ unsigned int sdata[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_count = 0;
    if (idx < size) {
        T val = a_ptr[idx];
        if (val != static_cast<T>(0)) {
            local_count = 1;
        }
    }
    sdata[tid] = local_count;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(result, (unsigned long long)sdata[0]);
    }
}


size_t NonZeroImpl<Device::CUDA>::execute(const Tensor& a) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(),a.data());
    // 分配设备内存并初始化为0
    unsigned long long* d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));
    cudaMemset(d_result, 0, sizeof(unsigned long long));
    std::visit([&](auto A_ptr){
        size_t shared = threads * sizeof(unsigned int);
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(A_ptr)>>; // const T* --> const T --> T
        if constexpr(std::is_same_v<AType,float16>){
            non_zero_cuda<<<blocks, threads, shared, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()),numel,d_result);
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            non_zero_cuda<<<blocks, threads, shared, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()),numel,d_result);
        }else{
            non_zero_cuda<AType><<<blocks, threads, shared, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()),numel,d_result);
        }
        ctx_impl->wait();

    },A);
    unsigned long long h_result;
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    ctx_impl->wait();
    return static_cast<size_t>(h_result);
}
template struct EqualImpl<Device::CUDA>;
template struct NotEqualImpl<Device::CUDA>;
template struct GreaterImpl<Device::CUDA>;
template struct LessImpl<Device::CUDA>;
template struct GreaterEqualImpl<Device::CUDA>;
template struct LessEqualImpl<Device::CUDA>;
template struct NonZeroImpl<Device::CUDA>;
}