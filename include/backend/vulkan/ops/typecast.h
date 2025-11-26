#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <Device D> struct TypecastImpl;

template <>
struct TypecastImpl<Device::VULKAN>{
    static void execute(Tensor& a, DataType dst_type);

    static Tensor execute(const Tensor& a, DataType dst_type);
};

extern template struct TypecastImpl<Device::VULKAN>;


}