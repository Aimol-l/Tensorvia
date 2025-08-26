#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct SliceImpl;

template <>
struct SliceImpl<Device::CUDA>{
    static Tensor execute(const Tensor& a, const std::vector<std::pair<int, int>>& ranges);
};

extern template struct SliceImpl<Device::CUDA>;

}