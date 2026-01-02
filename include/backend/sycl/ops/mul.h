#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

#include <limits>

namespace ops {

//***************************************
template <via::Device D> struct MulImpl;
//***************************************

template <>
struct MulImpl<via::Device::SYCL> {
    // [w,h] @ [h,w] --> [w,w]
    // [b,w,h] @ [b,h,w] --> [b,w,w]
    static Tensor execute(const Tensor& a, const Tensor& b);
};

extern template struct MulImpl<via::Device::SYCL>;
}