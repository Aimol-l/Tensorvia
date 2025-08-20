#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <Device D>
struct MulImpl;

template <>
struct MulImpl<Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

extern template struct MulImpl<Device::CPU>;
}  // namespace ops