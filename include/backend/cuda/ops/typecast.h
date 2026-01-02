#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <via::Device D> struct TypecastImpl;

template <>
struct TypecastImpl<via::Device::CUDA>{
    static Tensor execute(const Tensor& a, via::DataType dst_type);
};

extern template struct TypecastImpl<via::Device::CUDA>;


}