#pragma once
#include <limits>
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

//************************************
template <via::Device D> struct ConcatImpl;
//************************************
template <>
struct ConcatImpl<via::Device::SYCL> {
    static Tensor execute(const std::vector<Tensor> &tensors, int dim);
};

extern template struct ConcatImpl<via::Device::SYCL>;
}