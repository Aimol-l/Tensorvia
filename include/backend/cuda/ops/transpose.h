#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct TransposeImpl;

template <>
struct TransposeImpl<Device::CUDA>{
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a,Tensor& dst,std::initializer_list<int64_t> axes);
};

extern template struct TransposeImpl<Device::CUDA>;
}