#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <via::Device D>
struct TypecastImpl;

template <>
struct TypecastImpl<via::Device::CPU> {
    static void execute(Tensor& a, via::DataType dst_type);
    static Tensor execute(const Tensor& a, via::DataType dst_type);
};

extern template struct TypecastImpl<via::Device::CPU>;

}  // namespace ops