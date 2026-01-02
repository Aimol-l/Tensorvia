#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <via::Device D>
struct ConcatImpl;
template <>
struct ConcatImpl<via::Device::VULKAN> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<via::Device::VULKAN>;
}