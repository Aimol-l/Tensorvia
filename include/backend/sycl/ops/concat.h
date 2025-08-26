#pragma once
#include <limits>
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

//************************************
template <Device D>
struct ConcatImpl;
//************************************
template <>
struct ConcatImpl<Device::SYCL> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<Device::SYCL>;
}