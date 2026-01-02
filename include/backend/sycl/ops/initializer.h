#pragma once
#include <cstddef>
#include <cstring>
#include <random>
#include <print>
#include <stdfloat>
#include <chrono>
#include <oneapi/dpl/random>

#include <sycl/sycl.hpp>
#include "core/ops.h"
#include "backend/sycl/sycl_context.h"

namespace ops {
    
template <via::Device D> struct ZerosImpl;
template <via::Device D> struct OnesImpl;
template <via::Device D> struct FillImpl;
template <via::Device D> struct RandomImpl;

template <>
struct ZerosImpl<via::Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct OnesImpl<via::Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct FillImpl<via::Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float value);
};

template <>
struct RandomImpl<via::Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float min, float max);
};

extern template struct ZerosImpl<via::Device::SYCL>;
extern template struct OnesImpl<via::Device::SYCL>;
extern template struct FillImpl<via::Device::SYCL>;
extern template struct RandomImpl<via::Device::SYCL>;


}