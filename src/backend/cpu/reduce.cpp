#include <print>
#include <cmath>
#include <numbers>

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
        bool res = true;
        for(size_t i = 0; i < n; ++i){
            if(ptr[i] != static_cast<T>(val)){
                res = false;
                break;
            }
        }
        return res;
    } else {
        // 浮点类型：允许误差的比较
        const T tolerance = static_cast<T>(1e-5); // 根据需求调整容差
        bool res = true;
        for(size_t i = 0; i < n; ++i){
            if(std::abs(ptr[i] - static_cast<T>(val)) > tolerance){
                res = false;
                break;
            }
        }
        return  res;
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
        bool res = false;
        for(size_t i = 0; i < n; ++i){
            if(ptr[i] == static_cast<T>(val)){
                res = true;
                break;
            }
        }
        return res;
    } else {

        // 浮点类型：允许误差的比较
        bool res = false;
        for(size_t i = 0; i < n; ++i){
            if(std::abs(ptr[i] - static_cast<T>(val)) <= tolerance){
                res = true;
                break;
            }
        }
        return res;
    }
}

// ================================================================
float SumImpl<Device::CPU>::execute(const Tensor& a) {
    float sum = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        sum = sum_kernel<T>(a_ptr,a.numel());
    });
    return sum;
}
float MinImpl<Device::CPU>::execute(const Tensor& a) {
    float min = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        min = min_kernel<T>(a_ptr,a.numel());
    });
    return min;
}
float MaxImpl<Device::CPU>::execute(const Tensor& a) {
    float max = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        max = max_kernel<T>(a_ptr,a.numel());
    });
    return max;
}
float MeanImpl<Device::CPU>::execute(const Tensor& a) {
    float mean = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        mean = mean_kernel<T>(a);
    });
    return mean;
}
// ================================================================
Tensor SumImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    auto a_shape = a.shape();
    int dim = a_shape.size();
    size_t outer_size = 1;
    for(int i = 0; i < axis; ++i) outer_size *= a_shape[i];
    size_t axis_size = a.shape(axis);
    size_t inner_size = 1;
    for(int i = axis+1; i < dim; ++i) inner_size *= a_shape[i];


    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        sum_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor MinImpl<Device::CPU>::execute(const Tensor& a, int axis) {
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
    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        min_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor MaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
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

    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        max_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor MeanImpl<Device::CPU>::execute(const Tensor& a, int axis) {
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

    Tensor result(new_shape, DataType::FLOAT32, Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        float* res_ptr = static_cast<float*>(result.data());
        mean_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
// ================================================================
bool AllImpl<Device::CPU>::execute(const Tensor& a, float value) {
    bool all;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        all = all_kernel<T>(a_ptr, value, a.numel());
    });
    return all;
}
bool AnyImpl<Device::CPU>::execute(const Tensor& a, float value) {
    bool any;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        any = any_kernel<T>(a_ptr, value, a.numel());
    });
    return any;
}
// ================================================================
Tensor ArgMaxImpl<Device::CPU>::execute(const Tensor &a, int axis){
    // 移除 a.shape(axis) 所在的轴
    auto a_shape = a.shape();
    int dim = a_shape.size();
    std::vector<int64_t> new_shape;
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
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        argmax_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
Tensor ArgMinImpl<Device::CPU>::execute(const Tensor &a, int axis) {
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

    Tensor result(new_shape, DataType::INT32, Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        T* res_ptr = static_cast<T*>(result.data());
        argmin_kernel<T>(a_ptr,res_ptr,outer_size, inner_size, axis_size);
    });
    return result;
}
// ================================================================
template struct SumImpl<Device::CPU>;
template struct MaxImpl<Device::CPU>;
template struct MinImpl<Device::CPU>;
template struct MeanImpl<Device::CPU>;
template struct ArgMaxImpl<Device::CPU>;
template struct ArgMinImpl<Device::CPU>;
template struct AllImpl<Device::CPU>;
template struct AnyImpl<Device::CPU>;
}