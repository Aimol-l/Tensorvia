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
    

template <Device D>
struct ZerosImpl;

template <Device D>
struct OnesImpl;

template <Device D>
struct FillImpl;

template <Device D>
struct RandomImpl;
// *****************************************

template <>
struct ZerosImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct OnesImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct FillImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
};

template <>
struct RandomImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, double min, double max);
};

extern template struct ZerosImpl<Device::SYCL>;
extern template struct OnesImpl<Device::SYCL>;
extern template struct FillImpl<Device::SYCL>;
extern template struct RandomImpl<Device::SYCL>;


}