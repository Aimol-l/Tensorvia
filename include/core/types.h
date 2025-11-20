#pragma once

#include <cstdint>
#include <type_traits>

// #ifndef BACKEND_CUDA
//   #define BACKEND_CUDA //临时使用
// #endif

template<typename T>
struct compute_type_helper { using type = T; };

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

  using float32 = float;
  using float64 = double;
#endif

#ifdef BACKEND_VULKAN
  #include <stdfloat>
  using float16 = std::float16_t;
  using bfloat16 = std::bfloat16_t;
  using float32 = float;
  using float64 = double;
#endif

#ifdef BACKEND_SYCL // 默认指用icpx 编译
  #include <sycl/sycl.hpp>
  using float16 = sycl::half;
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  using float32 = float;    // 标准单精度浮点
  using float64 = double;    // 标准双精度浮点
#endif

#ifdef BACKEND_CUDA // 默认指用nvcc 编译
  #include "cuda_fp16.h"
  #include "cuda_bf16.h"
  #include "backend/cuda/float16.hpp"
  #include "backend/cuda/bfloat16.hpp"

  using float16 = ops::float16_t;
  using bfloat16 = ops::bfloat16_t;
  using float32 = float;            // 标准单精度浮点
  using float64 = double;           // 标准双精度浮点

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