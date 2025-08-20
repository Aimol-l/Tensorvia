#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D>
struct ConcatImpl;
template <>
struct ConcatImpl<Device::CUDA> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<Device::CUDA>;
}