#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct TransposeImpl;

template <>
struct TransposeImpl<Device::CUDA>{
    static void execute(Tensor& a);
    static Tensor execute(Tensor& a, std::initializer_list<int> axes);
};

extern template struct TransposeImpl<Device::CUDA>;
}