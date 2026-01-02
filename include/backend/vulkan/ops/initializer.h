#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops { 

template <via::Device D> struct ZerosImpl;
template <via::Device D> struct OnesImpl;
template <via::Device D> struct FillImpl;
template <via::Device D> struct RandomImpl;

template <>
struct ZerosImpl<via::Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct OnesImpl<via::Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct FillImpl<via::Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float value);
};

template <>
struct RandomImpl<via::Device::VULKAN> { 
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype,float min,float max);
};

extern template struct ZerosImpl<via::Device::VULKAN>;
extern template struct OnesImpl<via::Device::VULKAN>;
extern template struct FillImpl<via::Device::VULKAN>;
extern template struct RandomImpl<via::Device::VULKAN>;

}