#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct MulImpl;

template <>
struct MulImpl<Device::CUDA> {
    // [w,k] @ [k,h] --> [w,h]
    // [b,w,k] @ [b,k,h] --> [b,w,h]
    static Tensor execute(const Tensor& a, const Tensor& b);
};

extern template struct MulImpl<Device::CUDA>;

}