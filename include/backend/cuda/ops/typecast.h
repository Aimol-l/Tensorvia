#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct TypecastImpl;

template <>
struct TypecastImpl<Device::CUDA>{
    static Tensor execute(const Tensor& a, DataType dst_type);
};

extern template struct TypecastImpl<Device::CUDA>;


}