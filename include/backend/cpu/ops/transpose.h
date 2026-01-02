#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <via::Device D>
struct TransposeImpl;

template <>
struct TransposeImpl<via::Device::CPU> {
    static void execute(Tensor& a);

    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a,Tensor& dst, std::initializer_list<int64_t> axes);

};

extern template struct TransposeImpl<via::Device::CPU>;
}  // namespace ops