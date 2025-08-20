#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

//************************************************
template <Device D>
struct TypecastImpl;
//************************************************

template <>
struct TypecastImpl<Device::SYCL>{
    static Tensor execute(const Tensor& a, DataType dst_type);
};

extern template struct TypecastImpl<Device::SYCL>;
}