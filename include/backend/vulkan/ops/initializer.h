#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops { 

template <Device D> struct ZerosImpl;
template <Device D> struct OnesImpl;
template <Device D> struct FillImpl;
template <Device D> struct RandomImpl;

template <>
struct ZerosImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct OnesImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct FillImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
};

template <>
struct RandomImpl<Device::VULKAN> { 
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype,float min,float max);
};

extern template struct ZerosImpl<Device::VULKAN>;
extern template struct OnesImpl<Device::VULKAN>;
extern template struct FillImpl<Device::VULKAN>;
extern template struct RandomImpl<Device::VULKAN>;

}