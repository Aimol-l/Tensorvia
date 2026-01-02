#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


//************************************************
template <via::Device D> struct TransposeImpl;
//************************************************

template <>
struct TransposeImpl<via::Device::SYCL>{
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a,Tensor& dst,std::initializer_list<int64_t> axes);
};

extern template struct TransposeImpl<via::Device::SYCL>;
}