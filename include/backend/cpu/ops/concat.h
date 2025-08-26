#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

//****************************************
template <Device D>
struct ConcatImpl;
//****************************************

template <>
struct ConcatImpl<Device::CPU> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<Device::CPU>;
}