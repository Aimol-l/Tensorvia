#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <via::Device D> struct TypecastImpl;

template <>
struct TypecastImpl<via::Device::VULKAN>{
    static void execute(Tensor& a, via::DataType dst_type);

    static Tensor execute(const Tensor& a, via::DataType dst_type);
};

extern template struct TypecastImpl<via::Device::VULKAN>;


}