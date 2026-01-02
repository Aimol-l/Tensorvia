#pragma once

#include <cstdint>
#include <type_traits>

// #define BACKEND_SYCL

template<typename T>
struct compute_type_helper { using type = T; };

// 定义通用的类型别名，不管使用哪个后端
using float32 = float;
using float64 = double;

// float16和bfloat16的定义需要更通用的方法
// 如果已经通过后端定义了这些类型，则使用已定义的
// 否则，提供一个通用定义（依赖于编译器支持）
#if defined(BACKEND_CPU) || defined(BACKEND_VULKAN) || defined(BACKEND_SYCL) || defined(BACKEND_CUDA)
    // 已经在后端定义过了，不做重复定义
#else
    // 未定义任何后端时，提供通用定义
    #if __cplusplus >= 202302L && !defined(_MSC_VER)
        // C++23 并且不是 MSVC，用标准库
        #include <stdfloat>
        using float16  = std::float16_t;
        using bfloat16 = std::bfloat16_t;
    #else
        // 对于其他情况，需要提供一个后备方案
        // 注意：这里我们只是声明类型，实际实现需要在库内部处理
        #ifdef __cplusplus
        extern "C" {
        #endif
            // 使用编译器支持的半精度类型（如果可用）
            #if defined(__F16C__) || (defined(__x86_64__) && defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9)))
                typedef _Float16 float16;
            #elif defined(__ARM_FP16_FORMAT_IEEE) && defined(__ARM_FP16_ARGS)
                typedef _Float16 float16;
            #else
                // 如果编译器不支持半精度类型，使用uint16_t作为占位符
                // 但实际的实现需要在库内部进行转换
                typedef uint16_t float16;
            #endif
            
            // 对于bfloat16，使用类似的方法
            #if defined(__AVX512BF16__)
                typedef __bf16 bfloat16;
            #else
                typedef uint16_t bfloat16;
            #endif
        #ifdef __cplusplus
        }
        #endif
    #endif
#endif

// 为各种后端定义特定的类型映射
#ifdef BACKEND_CPU
  #if __cplusplus >= 202302L && !defined(_MSC_VER)
    // C++23 并且不是 MSVC，用标准库
    #include <stdfloat>
    using float16  = std::float16_t;
    using bfloat16 = std::bfloat16_t;
  #else
    // C++20 或 MSVC，没有标准库 float16/bfloat16，就用自定义
    #include "backend/cuda/float16.hpp"
    #include "backend/cuda/bfloat16.hpp"
    using float16  = ops::float16_t;
    using bfloat16 = ops::bfloat16_t;
  #endif
#endif

#ifdef BACKEND_VULKAN
  #include <stdfloat>
  using float16 = std::float16_t;
  using bfloat16 = std::bfloat16_t;
#endif

#ifdef BACKEND_SYCL // 默认指用icpx 编译
  #include <sycl/sycl.hpp>
  using float16 = sycl::half;
  using bfloat16 = sycl::ext::oneapi::bfloat16;
#endif

#ifdef BACKEND_CUDA // 默认指用nvcc 编译
  #include "cuda_fp16.h"
  #include "cuda_bf16.h"
  #include "backend/cuda/float16.hpp"
  #include "backend/cuda/bfloat16.hpp"

  using float16 = ops::float16_t;
  using bfloat16 = ops::bfloat16_t;

  template<>
  struct compute_type_helper<__half> { using type = float; };
  template<>
  struct compute_type_helper<__nv_bfloat16> { using type = float; };

#endif

#ifndef RESTRICT
    #if defined(_MSC_VER)
        #define RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define RESTRICT __restrict__
    #else
        #define RESTRICT
    #endif
#endif


template<>
struct compute_type_helper<float16> { using type = float; };
template<>
struct compute_type_helper<bfloat16> { using type = float; };

template<typename T>
using compute_type_t = typename compute_type_helper<T>::type;