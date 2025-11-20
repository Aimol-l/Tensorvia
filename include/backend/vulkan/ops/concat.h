#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <Device D>
struct ConcatImpl;
template <>
struct ConcatImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<Device::VULKAN>;
}