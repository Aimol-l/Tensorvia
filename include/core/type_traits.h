#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <stdfloat>
#include <variant>
#include <set>
#include <utility> // std::pair
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>


// ANSI 颜色代码
#define COLOR_RESET  "\033[0m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_CYAN   "\033[36m"

// 日志基础宏（内部使用）
#define LOG_BASE(level, color, msg) do { \
    auto now = std::chrono::system_clock::now(); \
    std::time_t t = std::chrono::system_clock::to_time_t(now); \
    std::tm tm_info; \
    localtime_r(&t, &tm_info); \
    std::cerr << color << "[" << level << "] " \
              << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S") \
              << " [" << __FILE__ << ":" << __LINE__ << "]" \
              << " (" << __func__ << "): " << msg << COLOR_RESET << std::endl; \
} while (0)

// 公开使用的日志宏
#define LOG_INFO(msg)  LOG_BASE("INFO",  COLOR_CYAN,   msg)
#define LOG_WARN(msg)  LOG_BASE("WARN",  COLOR_YELLOW, msg)
#define LOG_ERROR(msg) LOG_BASE("ERROR", COLOR_RED,    msg)


#define RUNNING_TIME(expr) \
    do { \
        auto start = std::chrono::steady_clock::now(); \
        expr; \
        auto end = std::chrono::steady_clock::now(); \
        std::chrono::duration<double, std::milli> duration = end - start; \
        std::cout << "Execution time: " << duration.count() << "ms" << std::endl; \
    } while (0)
//**********************************************************
enum class Device {
    CPU,
    CUDA,
    SYCL,
    VULKAN
};

enum class DataType{
    INT8 = 0,   // char
    INT16,  // short
    INT32,  // int
    INT64,
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    FLOAT64
};
constexpr std::string_view device_to_string(Device d) {
    switch (d) {
        case Device::CPU:    return "CPU";
        case Device::CUDA:   return "CUDA";
        case Device::SYCL:   return "SYCL";
        case Device::VULKAN: return "VULKAN";
        default:             return "UNKNOWN";
    }
}
constexpr std::string_view dtype_to_string(DataType t) {
    switch (t) {
        case DataType::INT8:     return "INT8";
        case DataType::INT16:    return "INT16";
        case DataType::INT32:    return "INT32";
        case DataType::INT64:    return "INT64";
        case DataType::FLOAT16:  return "FLOAT16";
        case DataType::FLOAT32:  return "FLOAT32";
        case DataType::FLOAT64:  return "FLOAT64";
        case DataType::BFLOAT16: return "BFLOAT16";
        default:                 return "UNKNOWN";
    }
}
inline size_t calc_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::INT8:    return 1;
        case DataType::INT16:   return 2;
        case DataType::INT32:   return 4;
        case DataType::INT64:   return 8;
        case DataType::FLOAT16: return 2;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        case DataType::BFLOAT16:return 2;
        default: throw std::runtime_error("Unsupported DataType");
    }
}

// 定义合法的类型转换规则
static const std::set<std::pair<DataType, DataType>> valid_casts = {
    {DataType::INT8,   DataType::FLOAT32}, // 255 -> 255.0f
    {DataType::INT16,  DataType::FLOAT32}, // 65535 -> 65535.0f
    {DataType::INT16,  DataType::FLOAT64}, // 65535 -> 65535.0
    {DataType::INT32,  DataType::FLOAT32}, // 4294967295 -> 4294967295.0f
    {DataType::INT32,  DataType::FLOAT64}, // 4294967295 -> 4294967295.0
    {DataType::INT64,  DataType::FLOAT64}, // 18446744073709551615 -> 18446744073709551615.0
    {DataType::FLOAT32,DataType::INT32},   // 3.1415f -> 3
    {DataType::FLOAT32,DataType::INT64},   // 3.1415f -> 3
    {DataType::INT8,   DataType::INT16},   // 255 -> 255
    {DataType::INT8,   DataType::INT32},   // 255 -> 255
    {DataType::INT8,   DataType::INT64},   // 255 -> 255
    {DataType::INT16,  DataType::INT32},   // 65535 -> 65535
    {DataType::INT32,  DataType::INT64},   // 4294967295 -> 4294967295
    {DataType::FLOAT16,DataType::FLOAT32}, // 1.0h -> 1.0f
    {DataType::FLOAT16,DataType::FLOAT64}, // 1.0h -> 1.0
    {DataType::FLOAT16,DataType::BFLOAT16}, // 1.0h -> 1.0f -> 1.0b
    {DataType::BFLOAT16,DataType::FLOAT32}, // 1.0b -> 1.0f
    {DataType::BFLOAT16,DataType::FLOAT64}, // 1.0b -> 1.0
    {DataType::BFLOAT16,DataType::FLOAT16},  // 3.14b -> 3.14f -> 3.14h
    {DataType::FLOAT32,DataType::FLOAT64}, // 1.0f -> 1.0
    {DataType::FLOAT64,DataType::FLOAT32}, // 1.0 -> 1.0f  可能有问题！！
};
using CppDTypeVariant = std::variant<
    int8_t*, int16_t*, int32_t*, int64_t*,
    float16*, bfloat16*, float*, double*
>;
using ConstCppDTypeVariant = std::variant<
    const int8_t*, 
    const int16_t*,
    const int32_t*, 
    const int64_t*,
    const float16*, 
    const bfloat16*, 
    const float32*, 
    const float64*
>;
template <typename T>
ConstCppDTypeVariant make_variant(const void* src) {
    ConstCppDTypeVariant var;
    var.emplace<const T*>(static_cast<const T*>(src));
    return var;
}
inline ConstCppDTypeVariant data_as_const_variant(DataType type, const void* src) {
    switch (type) {
        case DataType::INT8:    
            return make_variant<int8_t>(src);
        case DataType::INT16:   
            return make_variant<int16_t>(src);
        case DataType::INT32:   
            return make_variant<int32_t>(src);
        case DataType::INT64:   
            return make_variant<int64_t>(src);
        case DataType::FLOAT16: 
            return make_variant<float16>(src);
        case DataType::BFLOAT16:
            return make_variant<bfloat16>(src);
        case DataType::FLOAT32: 
            return make_variant<float32>(src);
        case DataType::FLOAT64: 
            return make_variant<float64>(src);
        default:
            throw std::runtime_error("Unsupported dtype for variant");
    }
}
template<typename Func>
void dispatch_dtype(DataType type, Func&& f) {
    switch (type) {
        case DataType::INT8:     f(std::type_identity<int8_t>{}); break;
        case DataType::INT16:    f(std::type_identity<int16_t>{}); break;
        case DataType::INT32:    f(std::type_identity<int32_t>{}); break;
        case DataType::INT64:    f(std::type_identity<int64_t>{}); break;
        case DataType::FLOAT16:  f(std::type_identity<float16>{}); break;
        case DataType::BFLOAT16: f(std::type_identity<bfloat16>{}); break;
        case DataType::FLOAT32:  f(std::type_identity<float>{}); break;
        case DataType::FLOAT64:  f(std::type_identity<double>{}); break;
        default:
            throw std::runtime_error("Unsupported dtype");
    }
}

// int8*int8 -> int8; int8*int16 -> int16; int8*int32 -> int32;int8*int64 -> int64; int8*float16 -> float16; int8*bfloat16 -> bfloat16; int8*float32 -> float32; int8*float64 -> float64;
// int16*int16 -> int16; int16*int32 -> int32; int16*int64 -> int64; int16*float16 -> float16; int16*bfloat16 -> bfloat16; int16*float32 -> float32; int16*float64 -> float64;
// int32*int32 -> int32; int32*int64 -> int64; int32*float16 -> float16; int32*bfloat16 -> bfloat16; int32*float32 -> float32; int32*float64 -> float64;
// int64*int64 -> int64; int64*float16 -> float64; int64*bfloat16 -> float64; int64*float32 -> float64; int64*float64 -> float64;
// float16*float16 -> float16; float16*bfloat16 -> float32; float16*float32 -> float32; float16*float64 -> float64;
// bfloat16*bfloat16 -> bfloat16; bfloat16*float32 -> float32; bfloat16*float64 -> float64;
// float32*float32 -> float32; float32*float64 -> float64;
// float64*float64 -> float64;
inline DataType compute_type(DataType type0, DataType type1){
    if(type0 == type1)      
        return type0;
    if(type0 <= DataType::INT64 && type1 <= DataType::INT64){
        // 都是整数
        return std::max(type0,type1);
    }else if(type0 > DataType::INT64 && type1 > DataType::INT64){
        // 都是浮点数
        // float16 + float16 = float32; bfloat16 + bfloat16 = float32;
        return std::max(std::max(type0,DataType::FLOAT32),std::max(type1,DataType::FLOAT32));
    }else{
        // 整数 + 浮点数 or 浮点数 + 整数
        if(type0 == DataType::INT64 || type1 == DataType::INT64)
            return DataType::FLOAT64;
        return std::max(std::max(type0,DataType::FLOAT32),std::max(type1,DataType::FLOAT32));
    }
    return DataType::FLOAT32;
}

inline bool is_cast_valid(DataType src, DataType dst) {
    return src == dst || valid_casts.contains({src, dst});
}
// 返回元素个数
inline size_t calc_numel(const std::vector<int64_t>& shape) {
    size_t numel = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("Invalid tensor dimension");
        numel *= dim;
    }
    return numel;
}

// 返回每一维的字节 stride（从最内层维度开始是 dtype size）
inline std::vector<size_t> calc_strides(const std::vector<int64_t>& shape, DataType dtype) {
    std::vector<size_t> strides(shape.size());
    size_t stride = calc_dtype_size(dtype);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}