#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

//************************************************
template <via::Device D>
struct TypecastImpl;
//************************************************

template <>
struct TypecastImpl<via::Device::SYCL>{
    static Tensor execute(const Tensor& a, via::DataType dst_type);
};

extern template struct TypecastImpl<via::Device::SYCL>;
}