#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <via::Device D>
struct MulImpl;

template <>
struct MulImpl<via::Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

extern template struct MulImpl<via::Device::CPU>;
}  // namespace ops