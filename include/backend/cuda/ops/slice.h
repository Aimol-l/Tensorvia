#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <via::Device D> struct SliceImpl;

template <>
struct SliceImpl<via::Device::CUDA>{
    static Tensor execute(const Tensor& a, const std::vector<std::pair<int64_t, int64_t>>& ranges);
};

extern template struct SliceImpl<via::Device::CUDA>;

}