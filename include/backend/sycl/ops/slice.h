#pragma once
#include <memory>
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


//*********************************************
template <via::Device D> struct SliceImpl;
//*********************************************

template <>
struct SliceImpl<via::Device::SYCL>{
    static Tensor execute(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);
};
extern template struct SliceImpl<via::Device::SYCL>;
}