#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <Device D>
struct TypecastImpl;

template <>
struct TypecastImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, DataType dst_type);
};

extern template struct TypecastImpl<Device::CPU>;

}  // namespace ops