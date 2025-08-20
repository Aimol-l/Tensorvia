#pragma once
#include <memory>
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


//*********************************************
template <Device D>
struct SliceImpl;
//*********************************************

template <>
struct SliceImpl<Device::SYCL>{
    static Tensor execute(const Tensor& t, const std::vector<std::pair<int, int>>& ranges);
};
extern template struct SliceImpl<Device::SYCL>;
}