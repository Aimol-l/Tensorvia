#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

//****************************************
template <via::Device D>
struct ConcatImpl;
//****************************************

template <>
struct ConcatImpl<via::Device::CPU> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<via::Device::CPU>;
}