#include "backend/cuda/ops/reduce.h"

namespace ops {

template<typename T>
__global__ void sum_cuda(const T* a_ptr, float* res, size_t numel) {
    extern __shared__ float sum_sdata[];
    uint32_t tid = threadIdx.x;         // 局部线程id
    T sum = T(0);

    for(int i = tid; i < numel; i += blockDim.x * gridDim.x) {
        sum += a_ptr[i];
    }

    sum_sdata[tid] = sum;

    __syncthreads(); // 等待所有线程都完成运算，部分和都保存到了sdata中
    // 归约优化
    for (uint32_t s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) 
            sum_sdata[tid] += sum_sdata[tid + s];
        __syncthreads();
    }

    #pragma unroll
    for(int offset = 32; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sum_sdata[tid] += sum_sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) 
        res[blockIdx.x] = sum_sdata[0]; // 最后在外部对res进行求和就行
}

float SumImpl<Device::CUDA>::execute(const Tensor& a) { 
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    float* res;
    // 在设备上分配结果内存
    cudaMallocManaged(&res, sizeof(float) * blocks);
    cudaMemset(res, 0, sizeof(float) * blocks);
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto a_ptr){
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            sum_cuda<__half><<<blocks, threads, threads * sizeof(float), ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), res, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            sum_cuda<__nv_bfloat16><<<blocks, threads, threads * sizeof(float), ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), res, numel);
        }else{
            sum_cuda<AType><<<blocks, threads, threads * sizeof(float), ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), res, numel);
        }
    }, A);
    ctx_impl->wait();
    float sum = 0;

    for (size_t i = 0; i < blocks; ++i) {
        sum += res[i];
    }
    cudaFree(res);
    return sum;
}

// kernel: 累加 axis 上的和，输入类型 T，累加/输出类型 R
template <typename T, typename R = T>
__global__ void sum_reduce_cuda(
    const T* a_ptr,
    R* out_ptr,
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;

    const size_t outer_idx = id / inner_size;
    const size_t inner_idx = id % inner_size;
    const size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;

    R acc = R(0);

    // 手工展开（4-way），兼顾对齐与性能，不做危险的 reinterpret_cast
    constexpr int UNROLL = 4;

    // 常见情况： axis_size 可能很大，这里按步长 UNROLL 累加
    size_t i = 0;
    for (; i + (UNROLL - 1) < axis_size; i += UNROLL) {
        // 索引为 base_offset + (i + k) * inner_size
        // 对不同 T 做正确的临时转换再加到 acc
        if constexpr (std::is_same_v<T, int8_t>) {
            // promote int8 to int32 before accumulate
            int32_t v0 = static_cast<int32_t>(a_ptr[base_offset + (i + 0) * inner_size]);
            int32_t v1 = static_cast<int32_t>(a_ptr[base_offset + (i + 1) * inner_size]);
            int32_t v2 = static_cast<int32_t>(a_ptr[base_offset + (i + 2) * inner_size]);
            int32_t v3 = static_cast<int32_t>(a_ptr[base_offset + (i + 3) * inner_size]);
            acc += static_cast<R>(v0 + v1 + v2 + v3);
        } else if constexpr (std::is_same_v<T, __half>) {
            // __half -> float conversion
            float f0 = __half2float(a_ptr[base_offset + (i + 0) * inner_size]);
            float f1 = __half2float(a_ptr[base_offset + (i + 1) * inner_size]);
            float f2 = __half2float(a_ptr[base_offset + (i + 2) * inner_size]);
            float f3 = __half2float(a_ptr[base_offset + (i + 3) * inner_size]);
            acc += static_cast<R>(f0 + f1 + f2 + f3);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float f0 = __bfloat162float(a_ptr[base_offset + (i + 0) * inner_size]);
            float f1 = __bfloat162float(a_ptr[base_offset + (i + 1) * inner_size]);
            float f2 = __bfloat162float(a_ptr[base_offset + (i + 2) * inner_size]);
            float f3 = __bfloat162float(a_ptr[base_offset + (i + 3) * inner_size]);
            acc += static_cast<R>(f0 + f1 + f2 + f3);
        } else {
            // 默认路径：整型（>=16位）、float、double 等直接转换累加
            R v0 = static_cast<R>(a_ptr[base_offset + (i + 0) * inner_size]);
            R v1 = static_cast<R>(a_ptr[base_offset + (i + 1) * inner_size]);
            R v2 = static_cast<R>(a_ptr[base_offset + (i + 2) * inner_size]);
            R v3 = static_cast<R>(a_ptr[base_offset + (i + 3) * inner_size]);
            acc += (v0 + v1 + v2 + v3);
        }
    }
    // 处理剩余尾数
    for (; i < axis_size; ++i) {
        if constexpr (std::is_same_v<T, int8_t>) {
            acc += static_cast<R>(static_cast<int32_t>(a_ptr[base_offset + i * inner_size]));
        } else if constexpr (std::is_same_v<T, __half>) {
            acc += static_cast<R>(__half2float(a_ptr[base_offset + i * inner_size]));
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            acc += static_cast<R>(__bfloat162float(a_ptr[base_offset + i * inner_size]));
        } else {
            acc += static_cast<R>(a_ptr[base_offset + i * inner_size]);
        }
    }
    out_ptr[id] = acc;
}


Tensor SumImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();

    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, a.dtype(), Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            sum_reduce_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            sum_reduce_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(result.data()), axis_size, outer_size, numel);
        }else{
            sum_reduce_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}


template<typename T>
__global__ void min_cuda(const T* a_ptr, float* partial, size_t size) {
    extern __shared__ float min_sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    // 直接用 if constexpr 初始化最大值
    T thread_min;
    if constexpr (std::is_same_v<T, float>) {
        // thread_min = std::numeric_limits<float32>::max(); // 不能在和函数中使用主机函数
        thread_min = 3.402823466e+38f; // FP32 最大值
    } else if constexpr (std::is_same_v<T, double>) {
        // thread_min = std::numeric_limits<float64>::max();
        thread_min = 1.7976931348623157e+308; // FP64 最大值
    } else if constexpr (std::is_same_v<T, int8_t>) {
        thread_min = INT8_MAX;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        thread_min = INT16_MAX;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        thread_min = INT32_MAX;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        thread_min = INT64_MAX;
    } else if constexpr (std::is_same_v<T, __half>) {
        thread_min = __half(65504.0f); // FP16 最大值
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        thread_min = __nv_bfloat16(3.38953139e38f); // BF16 最大值
    } else { // 实际上这里代码不报错是因为几乎走不到这个分支
        thread_min = std::numeric_limits<T>::max();
    }
    // 每个线程处理多个元素
    for (; i < size; i += blockDim.x * gridDim.x) {
        thread_min = thread_min < a_ptr[i] ? thread_min : a_ptr[i];
    }
    // 写到共享内存
    min_sdata[tid] = thread_min;
    __syncthreads();
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            min_sdata[tid] = min_sdata[tid] < min_sdata[tid + s] ? min_sdata[tid] : min_sdata[tid + s];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            min_sdata[tid] = min_sdata[tid] < min_sdata[tid + s] ? min_sdata[tid] : min_sdata[tid + s];
        }
        __syncthreads();
    }
    // 输出每个 block 的最小值
    if (tid == 0) {
        partial[blockIdx.x] = min_sdata[0];
    }
}

float MinImpl<Device::CUDA>::execute(const Tensor& a) { 
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    float* partial;
    cudaMallocManaged(&partial, blocks * sizeof(float));
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            min_cuda<__half><<<blocks, threads, threads * sizeof(__half), ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), partial, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            min_cuda<__nv_bfloat16><<<blocks, threads, threads * sizeof(__nv_bfloat16), ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), partial, numel);
        }else{
            min_cuda<AType><<<blocks, threads, threads * sizeof(AType), ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), partial, numel);
        }
    }, A);
    ctx_impl->wait();

    float min_val = std::numeric_limits<float>::max();
    for (size_t i = 0; i < blocks; ++i) {
        min_val = std::min(min_val, partial[i]);
    }
    cudaFree(partial);

    return min_val;
}

template <typename T>
__global__ void min_reduce_cuda(
    const T* a_ptr,
    T* out_ptr,
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;
    const size_t outer_idx = id / inner_size;
    const size_t inner_idx = id % inner_size;
    const size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;
    // 如果 axis_size 为 0，这里行为未定义，外部应保证 axis_size > 0
    // 初始化为第一个元素
    T cur0 = a_ptr[base_offset];
    // 对于 __half/__nv_bfloat16 用 float 做比较以避免精度或运算符问题
    if constexpr (std::is_same_v<T, __half>) {
        float m = __half2float(cur0);
        for (size_t i = 1; i < axis_size; ++i) {
            float v = __half2float(a_ptr[base_offset + i * inner_size]);
            if (v < m) m = v;
        }
        // 把 float 再转回 __half
        out_ptr[id] = __float2half(m);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // 注意：不同 CUDA 版本 bfloat16 的转换函数名可能不同
        float m = __bfloat162float(cur0);
        for (size_t i = 1; i < axis_size; ++i) {
            float v = __bfloat162float(a_ptr[base_offset + i * inner_size]);
            if (v < m) m = v;
        }
        out_ptr[id] = __float2bfloat16(m); // 若无此函数，需要替代
    } else {
        // 常规类型（int, float, double 等）直接比较
        T m = cur0;
        for (size_t i = 1; i < axis_size; ++i) {
            T v = a_ptr[base_offset + i * inner_size];
            if (v < m) m = v;
        }
        out_ptr[id] = m;
    }
}

Tensor MinImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, a.dtype(), Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            min_reduce_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            min_reduce_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(result.data()), axis_size, outer_size, numel);
        }else{
            min_reduce_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}

template<typename T>
__global__ void max_cuda(const T* a_ptr, float* partial, size_t size) {
    extern __shared__ float max_sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    // 直接用 if constexpr 初始化最大值
    T thread_max;
    if constexpr (std::is_same_v<T, float>) {
        thread_max = -3.402823466e+38f; // FP32 最大值
    } else if constexpr (std::is_same_v<T, double>) {
        thread_max =  -1.7976931348623157e+308; // FP64 最大值
    } else if constexpr (std::is_same_v<T, int8_t>) {
        thread_max = INT8_MIN;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        thread_max = INT16_MIN;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        thread_max = INT32_MIN;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        thread_max = INT64_MIN;
    } else if constexpr (std::is_same_v<T, __half>) {
        thread_max = __half(-65504.0f); // FP16 最大值
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        thread_max = __nv_bfloat16(-3.38953139e38f); // BF16 最大值
    } else { // 实际上这里代码不报错是因为几乎走不到这个分支
        thread_max = std::numeric_limits<T>::min();
    }
    // 每个线程处理多个元素
    for (; i < size; i += blockDim.x * gridDim.x) {
        thread_max = thread_max < a_ptr[i] ? thread_max : a_ptr[i];
    }
    // 写到共享内存
    max_sdata[tid] = thread_max;
    __syncthreads();
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            max_sdata[tid] = max_sdata[tid] < max_sdata[tid + s] ? max_sdata[tid] : max_sdata[tid + s];
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            max_sdata[tid] = max_sdata[tid] < max_sdata[tid + s] ? max_sdata[tid] : max_sdata[tid + s];
        }
        __syncthreads();
    }
    // 输出每个 block 的最小值
    if (tid == 0) {
        partial[blockIdx.x] = max_sdata[0];
    }
}

float MaxImpl<Device::CUDA>::execute(const Tensor& a) { 
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    float* partial;
    cudaMallocManaged(&partial, blocks * sizeof(float));
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            max_cuda<__half><<<blocks, threads, threads * sizeof(__half), ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), partial, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            max_cuda<__nv_bfloat16><<<blocks, threads, threads * sizeof(__nv_bfloat16), ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), partial, numel);
        }else{
            max_cuda<AType><<<blocks, threads, threads * sizeof(AType), ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), partial, numel);
        }
    }, A);
    ctx_impl->wait();

    float max_val = std::numeric_limits<float>::min();
    for (size_t i = 0; i < blocks; ++i) {
        max_val = std::max(max_val, partial[i]);
    }
    cudaFree(partial);

    return max_val;
}

template <typename T>
__global__ void max_reduce_cuda(
    const T* a_ptr,
    T* out_ptr,
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;
    const size_t outer_idx = id / inner_size;
    const size_t inner_idx = id % inner_size;
    const size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;
    // 如果 axis_size 为 0，这里行为未定义，外部应保证 axis_size > 0
    // 初始化为第一个元素
    T cur0 = a_ptr[base_offset];
    // 对于 __half/__nv_bfloat16 用 float 做比较以避免精度或运算符问题
    if constexpr (std::is_same_v<T, __half>) {
        float m = __half2float(cur0);
        for (size_t i = 1; i < axis_size; ++i) {
            float v = __half2float(a_ptr[base_offset + i * inner_size]);
            if (v > m) m = v;
        }
        // 把 float 再转回 __half
        out_ptr[id] = __float2half(m);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // 注意：不同 CUDA 版本 bfloat16 的转换函数名可能不同
        float m = __bfloat162float(cur0);
        for (size_t i = 1; i < axis_size; ++i) {
            float v = __bfloat162float(a_ptr[base_offset + i * inner_size]);
            if (v > m) m = v;
        }
        out_ptr[id] = __float2bfloat16(m); // 若无此函数，需要替代
    } else {
        // 常规类型（int, float, double 等）直接比较
        T m = cur0;
        for (size_t i = 1; i < axis_size; ++i) {
            T v = a_ptr[base_offset + i * inner_size];
            if (v > m) m = v;
        }
        out_ptr[id] = m;
    }
}

Tensor MaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, a.dtype(), Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            max_reduce_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            max_reduce_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(result.data()), axis_size, outer_size, numel);
        }else{
            max_reduce_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}


float MeanImpl<Device::CUDA>::execute(const Tensor& a) { 
    Tensor a_copy = a.clone();
    return SumImpl<Device::CUDA>::execute(a_copy) / a_copy.numel(); // 直接复用
}

template <typename T, typename R = T>
__global__ void mean_reduce_cuda(
    const T* a_ptr,
    R* out_ptr,  // 输出类型可指定（如整数输入时用float输出）
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;

    const size_t outer_idx = id / inner_size;
    const size_t inner_idx = id % inner_size;
    const size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;

    // --- 核心修改点：将索引追踪改为累加 ---
    if constexpr (std::is_same_v<T, __half>) {
        // 半精度浮点：用float累加
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            sum += __half2float(a_ptr[base_offset + i * inner_size]); // 统一转换为float
        }
        out_ptr[id] = static_cast<R>(sum / axis_size);
    }else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // 半精度浮点：用float累加
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            sum += __bfloat162float(a_ptr[base_offset + i * inner_size]); // 统一转换为float
        }
        out_ptr[id] = static_cast<R>(sum / axis_size);
    }
    else if constexpr (std::is_integral_v<T>) {
        // 整数类型：必须用float计算避免溢出
        float sum = 0.0f;
        for (size_t i = 0; i < axis_size; ++i) {
            sum += static_cast<float>(a_ptr[base_offset + i * inner_size]);
        }
        out_ptr[id] = static_cast<R>(sum / axis_size + 0.5f); // 四舍五入
    }
    else {
        // 常规浮点类型（float/double）
        R sum = a_ptr[base_offset];
        for (size_t i = 1; i < axis_size; ++i) {
            sum += a_ptr[base_offset + i * inner_size];
        }
        out_ptr[id] = sum / static_cast<R>(axis_size);
    }
}

Tensor MeanImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, a.dtype(), Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            mean_reduce_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            mean_reduce_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(result.data()), axis_size, outer_size, numel);
        }else{
            mean_reduce_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}

template <typename T>
__global__ void argmax_cuda(
    const T* a_ptr,
    int32_t* out_ptr,  // index 输出
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;

    size_t outer_idx = id / inner_size;
    size_t inner_idx = id % inner_size;
    size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;

    T max_val = a_ptr[base_offset];
    int64_t max_idx = 0;
    for (size_t i = 1; i < axis_size; ++i) {
        T val = a_ptr[base_offset + i * inner_size];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    out_ptr[id] = max_idx;
}

Tensor ArgMaxImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, DataType::INT32, Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            argmax_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            argmax_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }else{
            argmax_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}


template <typename T>
__global__ void argmin_cuda(
    const T* a_ptr,
    int32_t* out_ptr,
    size_t axis_size,
    size_t inner_size,
    size_t numel)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;

    size_t outer_idx = id / inner_size;
    size_t inner_idx = id % inner_size;
    size_t base_offset = outer_idx * axis_size * inner_size + inner_idx;

    T min_val = a_ptr[base_offset];
    int64_t min_idx = 0;
    for (size_t i = 1; i < axis_size; ++i) {
        T val = a_ptr[base_offset + i * inner_size];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }
    out_ptr[id] = min_idx;
}

Tensor ArgMinImpl<Device::CUDA>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];
    
    Tensor result(new_shape, DataType::INT32, Device::CUDA);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(result.dtype(), result.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            argmin_cuda<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            argmin_cuda<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }else{
            argmin_cuda<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<int32_t*>(result.data()), axis_size, outer_size, numel);
        }
    }, A);

    ctx_impl->wait();

    return result;
}

template <typename T>
__global__ void any_cuda(const T* a_ptr, float val, size_t size, int* result) {
    // 声明共享内存：用于存放每个线程的比较结果
    extern __shared__ bool any_sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i   = blockIdx.x * blockDim.x + tid;
    // Step 1: 每个线程检查自己负责的元素是否等于 val
    bool equal = false;
    if (i < size) {
        if constexpr (std::is_floating_point_v<T>) {
            // 浮点数使用容差比较
            float diff = static_cast<float>(a_ptr[i] - val);
            equal = (fabsf(diff) < 1e-6f);
        } else if constexpr (std::is_same_v<T, float16>) {
            float diff = static_cast<float>(_half2float(a_ptr[i]) - val);
            equal = (fabsf(diff) < 1e-6f);
        } else if constexpr (std::is_same_v<T, bfloat16>) {
            float diff = static_cast<float>(_bfloat162float(a_ptr[i]) - val);
            equal = (fabsf(diff) < 1e-6f);
        }else {
            // 整型直接比较
            equal = (static_cast<float>(a_ptr[i]) == val);
        }
    }
    // Step 2: 写入共享内存
    any_sdata[tid] = equal; //将所有线程的比较结果写入共享内存
    __syncthreads(); // 等待所有线程工作完

    // Step 3: 并行规约 —— 使用逻辑 OR 合并所有结果
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            any_sdata[tid] = any_sdata[tid] || any_sdata[tid + stride];
        }
        __syncthreads();  // 每次合并后同步
    }
    // Step 4: block 内第 0 个线程将本 block 的结果写回全局 result
    if (tid == 0 && any_sdata[0]) {
        atomicExch((int*)result, 1);  // 原子地设置 result = true
    }
}

bool AnyImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    
    // 没有bool指针的情况下，使用int指针代替
    int* res;
    cudaMallocManaged(&res, sizeof(int));
    *res = 0;

    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            any_cuda<__half><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), val, numel, res);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            any_cuda<__nv_bfloat16><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), val, numel, res);
        }else{
            any_cuda<AType><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), val, numel, res);
        }
    }, A);
    ctx_impl->wait();
    bool result = *res;
    cudaFree(res);
    return result;

}

template <typename T>
__global__ void all_cuda(const T* a_ptr, float val, size_t size, int* result) {
    // 声明共享内存（用于 block 内规约）
    extern __shared__ bool all_sdata[];
    auto tid = threadIdx.x;
    auto i   = blockIdx.x * blockDim.x + threadIdx.x;
    // Step 1: 每个线程判断 a[i] == val
    bool local_result = true;  // 默认为 true（AND 的单位元）
    if (i < size) {
        if constexpr (std::is_floating_point_v<T>) {
            // 浮点数使用容差比较
            float diff = static_cast<float>(a_ptr[i] - val);
            local_result = (fabsf(diff) < 1e-6f);
        } else if constexpr (std::is_same_v<T, float16>) {
            float diff = static_cast<float>(_half2float(a_ptr[i]) - val);
            local_result = (fabsf(diff) < 1e-6f);
        } else if constexpr (std::is_same_v<T, bfloat16>) {
            float diff = static_cast<float>(_bfloat162float(a_ptr[i]) - val);
            local_result = (fabsf(diff) < 1e-6f);
        } else {
            // 整型直接比较
            local_result = (static_cast<float>(a_ptr[i]) == val);
        }
    }
    // 如果 i >= size（越界），保持 local_result = true（不影响 AND）
    // Step 2: 写入共享内存
    all_sdata[tid] = local_result;
    __syncthreads();
    // Step 3: 并行规约 —— 使用逻辑 AND (&&)
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            all_sdata[tid] = all_sdata[tid] && all_sdata[tid + stride];
        }
        __syncthreads();
    }
    // Step 4: block 内第 0 个线程将结果写回全局内存
    if (tid == 0) {
        // 只有当本 block 所有元素都满足条件时，sdata[0] 才为 true
        // 我们需要对所有 block 的结果做 AND：只要有一个 block 是 false，整体就是 false
        if (!all_sdata[0]) {
            atomicExch(result, 0);  // 原子地设置 result = 0
        }
        // 注意：我们不处理 any_sdata[0] == true 的情况，因为 result 初始应为 1
    }
}

bool AllImpl<Device::CUDA>::execute(const Tensor& a, float val) {
    size_t numel = a.numel();
    constexpr size_t threads = 256;
    size_t blocks = (numel + threads - 1) / threads;

    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    
    // 没有bool指针的情况下，使用int指针代替
    int* res;
    cudaMallocManaged(&res, sizeof(int));
    *res = 1;

    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr (std::is_same_v<AType, float16>) {
            all_cuda<__half><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), val, numel, res);
        }else if constexpr (std::is_same_v<AType, bfloat16>) {
            all_cuda<__nv_bfloat16><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), val, numel, res);
        }else{
            all_cuda<AType><<<blocks, threads, threads * sizeof(int), ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), val, numel, res);
        }
    }, A);
    ctx_impl->wait();
    bool result = *res;
    cudaFree(res);
    return result;
}


template struct SumImpl<Device::CUDA>;
template struct MinImpl<Device::CUDA>;
template struct MaxImpl<Device::CUDA>;   
template struct MeanImpl<Device::CUDA>;
template struct ArgMaxImpl<Device::CUDA>;
template struct ArgMinImpl<Device::CUDA>;
template struct AnyImpl<Device::CUDA>;
template struct AllImpl<Device::CUDA>;

}