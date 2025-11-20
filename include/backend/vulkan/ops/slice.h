#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <Device D> struct SliceImpl;

template <>
struct SliceImpl<Device::VULKAN>{
    static Tensor execute(const Tensor& a, const std::vector<std::pair<int64_t, int64_t>>& ranges);
};

extern template struct SliceImpl<Device::VULKAN>;

}