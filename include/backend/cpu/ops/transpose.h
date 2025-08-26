#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <Device D>
struct TransposeImpl;

template <>
struct TransposeImpl<Device::CPU> {
    static void execute(Tensor& a);

    static Tensor execute(Tensor& a, std::initializer_list<int> axes);
};

extern template struct TransposeImpl<Device::CPU>;
}  // namespace ops