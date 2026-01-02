#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <via::Device D> struct MulImpl;

template <>
struct MulImpl<via::Device::VULKAN> {
    // [w,k] @ [k,h] --> [w,h]
    // [b,w,k] @ [b,k,h] --> [b,w,h]
    static Tensor execute(const Tensor& a, const Tensor& b);
};

extern template struct MulImpl<via::Device::VULKAN>;

}