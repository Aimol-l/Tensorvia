#include <print>
#include <cmath>
#include <numbers>
#include <execution>

#include "core/tensor.h"
#include "core/types.h"
#include "backend/cpu/ops/reduce.h"

namespace ops {
template <typename T>
void argmax_kernel(const T *src_ptr, T *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T max_val = src_ptr[start];
        int32_t max_index = 0;

        // 在axis_size维度上寻找最小值索引
        for (size_t k = 1; k < axis_size; ++k) {
            const size_t pos = start + k * inner_size;
            if (src_ptr[pos] > max_val) {
                max_val = src_ptr[pos];
                max_index = k;
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = max_index;
    }
}

template <typename T>
void argmin_kernel(const T *src_ptr, T *res_ptr, int outer_size, int inner_size, int axis_size) {
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T min_val = src_ptr[start];
        int32_t min_index = 0;

        // 在axis_size维度上寻找最小值索引
        for (size_t k = 1; k < axis_size; ++k) {
            const size_t pos = start + k * inner_size;
            if (src_ptr[pos] < min_val) {
                min_val = src_ptr[pos];
                min_index = k;
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = min_index;
    }
}

template <typename T>
float sum_kernel(const T *ptr, size_t size) {
    float sum = 0.0f;
    #pragma omp parallel for simd reduction(+:sum) schedule(static) 
    for (size_t i = 0; i < size; ++i) {
        sum += static_cast<float>(ptr[i]);
    }
    return sum;
}

template <typename T>
float mean_kernel(const Tensor& src) {
    float total = sum_kernel<T>(static_cast<const T*>(src.data()), src.numel());  // 复用 sum_kernel
    return total / static_cast<float>(src.numel());
}

template <typename T>
float min_kernel(const T *ptr, size_t size) {

    float min_val = std::numeric_limits<float>::max();

    #pragma omp simd reduction(min:min_val)
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(ptr[i]);
        min_val = std::min(min_val, val);
    }

    return min_val;
}

template <typename T>
float max_kernel(const T *ptr, size_t size) {
    float max_val = std::numeric_limits<float>::lowest();
    #pragma omp simd reduction(max:max_val)
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(ptr[i]);
        max_val = std::max(max_val, val);
    }
    return max_val;
}

template <typename T>
void sum_kernel(const T *src_ptr, T *res_ptr, int outer_size, int inner_size, int axis_size){
    
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;
    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        float s = 0.0f;
        // 在axis_size维度上寻找最小值索引
        #pragma omp parallel for simd reduction(+:s) schedule(static) 
        for (size_t k = 0; k < axis_size; ++k) {
            const size_t pos = start + k * inner_size;
            s += static_cast<float>(src_ptr[pos]);
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = T(s);
    }
}

template <typename T>
void mean_kernel(const T *src_ptr, float *res_ptr, int outer_size, int inner_size, int axis_size){
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;
    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;

        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        float s = 0.0f;

        // 在axis_size维度上计算平均值
        #pragma omp parallel for simd reduction(+:s) schedule(static) 
        for (size_t k = 0; k < axis_size; ++k) {
            const size_t pos = start + k * inner_size;
            s += static_cast<float>(src_ptr[pos]);
        }
        s /= static_cast<float>(axis_size);

        res_ptr[task_id] = s;
    }
};

template <typename T>
void min_kernel(const T *src_ptr, T *res_ptr, int outer_size, int inner_size, int axis_size){
    
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T min_val = src_ptr[start];
        // 在axis_size维度上寻找最小值索引
        if constexpr (std::is_arithmetic_v<T>) {
            // 算术类型：使用 OpenMP SIMD 向量化优化
            #pragma omp simd reduction(min:min_val)
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                min_val = std::min(min_val, src_ptr[pos]);
            }
        } else {
            // 非算术类型：使用标准循环
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                min_val = std::min(min_val, src_ptr[pos]);
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = min_val;
    }
}

template <typename T>
void max_kernel(const T *src_ptr, T *res_ptr, int outer_size, int inner_size, int axis_size){
    
    const size_t total_tasks = static_cast<size_t>(outer_size) * inner_size;

    #pragma omp parallel for
    for (size_t task_id = 0; task_id < total_tasks; ++task_id) {
        // 计算当前任务对应的outer和inner索引
        const size_t outer = task_id / inner_size;
        const size_t inner = task_id % inner_size;
        
        // 计算当前组的起始内存位置
        const size_t start = outer * axis_size * inner_size + inner;
        T max_val = src_ptr[start];

        // 在axis_size维度上寻找最小值索引
        if constexpr (std::is_arithmetic_v<T>) {
            #pragma omp simd reduction(max:max_val)
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                max_val = std::max(max_val, src_ptr[pos]);
            }
        }else{
                // 非算术类型：使用标准循环
            for (size_t k = 0; k < axis_size; ++k) {
                const size_t pos = start + k * inner_size;
                max_val = std::min(max_val, src_ptr[pos]);
            }
        }
        // 直接写入对应的结果位置
        res_ptr[task_id] = max_val;
    }
}

template <typename T>
bool all_kernel(const T *ptr, float val, size_t n) {
    if constexpr (std::is_integral_v<T>) {
        // 整数类型：精确比较
        return std::all_of(std::execution::par, ptr, ptr + n, 
            [val](T x) { return x == static_cast<T>(val); });
    } else {
        // 浮点类型：允许误差的比较
        const T tolerance = static_cast<T>(1e-5); // 根据需求调整容差
        return std::all_of(std::execution::par, ptr, ptr + n,
            [val, tolerance](T x) { return std::abs(x - static_cast<T>(val)) <= tolerance; });
    }
}
template <typename T>
bool any_kernel(const T *ptr, float val, size_t n) {

    constexpr auto tolerance = []{
        if constexpr (std::is_same_v<T, float>) return 1e-5f;
        else if constexpr (std::is_same_v<T, float64>) return 1e-10;
        else if constexpr (std::is_same_v<T, float16>) return 1e-3f;
        else if constexpr (std::is_same_v<T, bfloat16>) return 1e-2f;
        else return T{0}; // 整数类型
    }();

    if constexpr (std::is_integral_v<T>) {
        // 整数类型：精确比较
        return std::any_of(std::execution::par, ptr, ptr + n,
            [val](T x) { return x == static_cast<T>(val); });
    } else {
        // 浮点类型：允许误差的比较
        return std::any_of(std::execution::par, ptr, ptr + n,[val, tolerance](T x) { return std::abs(x - static_cast<T>(val)) <= tolerance; });
    }
}

// ************************************** 特化 **************************************
float SumImpl<Device::CPU>::execute(const Tensor& a) {
    switch (a.dtype()) {
        case DataType::INT8:     return sum_kernel<int8_t>(static_cast<const int8_t*>(a.data()), a.numel());
        case DataType::INT16:    return sum_kernel<int16_t>(static_cast<const int16_t*>(a.data()), a.numel());
        case DataType::INT32:    return sum_kernel<int32_t>(static_cast<const int32_t*>(a.data()), a.numel());
        case DataType::INT64:    return sum_kernel<int64_t>(static_cast<const int64_t*>(a.data()), a.numel());
        case DataType::FLOAT16:  return sum_kernel<float16>(static_cast<const float16*>(a.data()), a.numel());
        case DataType::BFLOAT16: return sum_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), a.numel());
        case DataType::FLOAT32:  return sum_kernel<float32>(static_cast<const float32*>(a.data()), a.numel());
        case DataType::FLOAT64:  return sum_kernel<float64>(static_cast<const float64*>(a.data()), a.numel());
        default: throw std::runtime_error("sum: unsupported data type");
    }
}

Tensor SumImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int> new_shape;
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

    Tensor result(new_shape, a.dtype(), Device::CPU);

    switch (a.dtype()) {
        case DataType::INT8:      sum_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT16:     sum_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT32:     sum_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT64:     sum_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT16:   sum_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::BFLOAT16:  sum_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT32:   sum_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT64:   sum_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(result.data()), outer_size, inner_size, axis_size);break;
        default: throw std::runtime_error("sum: unsupported data type");
    }
    return result;
}
float MinImpl<Device::CPU>::execute(const Tensor& a) {
    switch (a.dtype()) {
        case DataType::INT8:     return min_kernel<int8_t>(static_cast<const int8_t*>(a.data()), a.numel());
        case DataType::INT16:    return min_kernel<int16_t>(static_cast<const int16_t*>(a.data()), a.numel());
        case DataType::INT32:    return min_kernel<int32_t>(static_cast<const int32_t*>(a.data()), a.numel());
        case DataType::INT64:    return min_kernel<int64_t>(static_cast<const int64_t*>(a.data()), a.numel());
        case DataType::FLOAT16:  return min_kernel<float16>(static_cast<const float16*>(a.data()), a.numel());
        case DataType::BFLOAT16: return min_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), a.numel());
        case DataType::FLOAT32:  return min_kernel<float32>(static_cast<const float32*>(a.data()), a.numel());
        case DataType::FLOAT64:  return min_kernel<float64>(static_cast<const float64*>(a.data()), a.numel());
        default: throw std::runtime_error("min: unsupported data type");
    }
}

Tensor MinImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int> new_shape;
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

    Tensor result(new_shape, a.dtype(), Device::CPU);

    switch (a.dtype()) {
        case DataType::INT8:     min_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT16:    min_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT32:    min_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT64:    min_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT16:  min_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::BFLOAT16: min_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT32:  min_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT64:  min_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(result.data()), outer_size, inner_size, axis_size);break;
        default: throw std::runtime_error("min: unsupported data type");
    }
    return result;
}
float MaxImpl<Device::CPU>::execute(const Tensor& a) {
    switch (a.dtype()) {
        case DataType::INT8:     return max_kernel<int8_t>(static_cast<const int8_t*>(a.data()), a.numel());
        case DataType::INT16:    return max_kernel<int16_t>(static_cast<const int16_t*>(a.data()), a.numel());
        case DataType::INT32:    return max_kernel<int32_t>(static_cast<const int32_t*>(a.data()), a.numel());
        case DataType::INT64:    return max_kernel<int64_t>(static_cast<const int64_t*>(a.data()), a.numel());
        case DataType::FLOAT16:  return max_kernel<float16>(static_cast<const float16*>(a.data()), a.numel());
        case DataType::BFLOAT16: return max_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), a.numel());
        case DataType::FLOAT32:  return max_kernel<float32>(static_cast<const float32*>(a.data()), a.numel());
        case DataType::FLOAT64:  return max_kernel<float64>(static_cast<const float64*>(a.data()), a.numel());
        default: throw std::runtime_error("max: unsupported data type");
    }
}

Tensor MaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int> new_shape;
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

    Tensor result(new_shape, a.dtype(), Device::CPU);

    switch (a.dtype()) {
        case DataType::INT8:     max_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT16:    max_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT32:    max_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT64:    max_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT16:  max_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::BFLOAT16: max_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT32:  max_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT64:  max_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(result.data()), outer_size, inner_size, axis_size);break;
        default: throw std::runtime_error("max: unsupported data type");
    }
    return result;
}
float MeanImpl<Device::CPU>::execute(const Tensor& a) {
    switch (a.dtype()) {
        case DataType::INT8:     return mean_kernel<int8_t>(a);
        case DataType::INT16:    return mean_kernel<int16_t>(a);
        case DataType::INT32:    return mean_kernel<int32_t>(a);
        case DataType::INT64:    return mean_kernel<int64_t>(a);
        case DataType::FLOAT16:  return mean_kernel<float16>(a);
        case DataType::BFLOAT16: return mean_kernel<bfloat16>(a);
        case DataType::FLOAT32:  return mean_kernel<float32>(a);
        case DataType::FLOAT64:  return mean_kernel<float64>(a);
        default: throw std::runtime_error("mean: unsupported data type");
    }
}
Tensor MeanImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int> new_shape;
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

    Tensor result(new_shape, DataType::FLOAT32, Device::CPU);

    switch (a.dtype()) {
        case DataType::INT8:     mean_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT16:    mean_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT32:    mean_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::INT64:    mean_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT16:  mean_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::BFLOAT16: mean_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT32:  mean_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        case DataType::FLOAT64:  mean_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float*>(result.data()), outer_size, inner_size, axis_size);break;
        default: throw std::runtime_error("mean: unsupported data type");
    }
    return result;
}
Tensor ArgMaxImpl<Device::CPU>::execute(const Tensor &a, int axis){
    // 移除 a.shape(axis) 所在的轴
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    // 依旧需要计算 坐标 映射的函数
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a.shape()[i];
    size_t axis_size = a.shape()[axis];
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a.shape()[i];
    
    Tensor result(new_shape,DataType::INT32,Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:     argmax_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT16:    argmax_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT32:    argmax_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT64:    argmax_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT16:  argmax_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT32:  argmax_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT64:  argmax_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::BFLOAT16: argmax_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(result.data()), outer_size, inner_size, axis_size); break;
        default: throw std::runtime_error("argmax: unsupported data type");
    }
    return result;
}
Tensor ArgMinImpl<Device::CPU>::execute(const Tensor &a, int axis) {
    std::vector<int> new_shape;
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

    Tensor result(new_shape, DataType::INT32, Device::CPU);
    switch (a.dtype()) {
        case DataType::INT8:     argmin_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<int8_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT16:    argmin_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<int16_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT32:    argmin_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<int32_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::INT64:    argmin_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<int64_t*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT16:  argmin_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT32:  argmin_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::FLOAT64:  argmin_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(result.data()), outer_size, inner_size, axis_size); break;
        case DataType::BFLOAT16: argmin_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(result.data()), outer_size, inner_size, axis_size); break;
        default: throw std::runtime_error("argmin: unsupported data type");
    }
    return result;
}
bool AllImpl<Device::CPU>::execute(const Tensor& a, float value) {
    switch (a.dtype()) {
        case DataType::INT8:     return all_kernel<int8_t>(static_cast<const int8_t*>(a.data()), value, a.numel());
        case DataType::INT16:    return all_kernel<int16_t>(static_cast<const int16_t*>(a.data()), value, a.numel());
        case DataType::INT32:    return all_kernel<int32_t>(static_cast<const int32_t*>(a.data()), value, a.numel());
        case DataType::INT64:    return all_kernel<int64_t>(static_cast<const int64_t*>(a.data()), value, a.numel());
        case DataType::FLOAT16:  return all_kernel<float16>(static_cast<const float16*>(a.data()), value, a.numel());
        case DataType::BFLOAT16: return all_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), value, a.numel());
        case DataType::FLOAT32:  return all_kernel<float32>(static_cast<const float32*>(a.data()), value, a.numel());
        case DataType::FLOAT64:  return all_kernel<float64>(static_cast<const float64*>(a.data()), value, a.numel());
        default: throw std::runtime_error("mean: unsupported data type");
    }
}
bool AnyImpl<Device::CPU>::execute(const Tensor& a, float value) {
    switch (a.dtype()) {
        case DataType::INT8:
            return any_kernel<int8_t>(static_cast<const int8_t *>(a.data()), value, a.numel());
        case DataType::INT16:
            return any_kernel<int16_t>(static_cast<const int16_t *>(a.data()), value, a.numel());
        case DataType::INT32:    return any_kernel<int32_t>(static_cast<const int32_t*>(a.data()), value, a.numel());
        case DataType::INT64:    return any_kernel<int64_t>(static_cast<const int64_t*>(a.data()), value, a.numel());
        case DataType::FLOAT16:  return any_kernel<float16>(static_cast<const float16*>(a.data()), value, a.numel());
        case DataType::BFLOAT16: return any_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), value, a.numel());
        case DataType::FLOAT32:  return any_kernel<float32>(static_cast<const float32*>(a.data()), value, a.numel());
        case DataType::FLOAT64:  return any_kernel<float64>(static_cast<const float64*>(a.data()), value, a.numel());
        default: throw std::runtime_error("mean: unsupported data type");
    }
}
template struct SumImpl<Device::CPU>;
template struct MaxImpl<Device::CPU>;
template struct MinImpl<Device::CPU>;
template struct MeanImpl<Device::CPU>;
template struct ArgMaxImpl<Device::CPU>;
template struct ArgMinImpl<Device::CPU>;
template struct AllImpl<Device::CPU>;
template struct AnyImpl<Device::CPU>;


}