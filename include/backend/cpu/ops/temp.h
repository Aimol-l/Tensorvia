#pragma once
#include "ops.h"
#include "backend/cpu/cpu_tensor.h"


namespace ops {

template <via::Device D> struct TempImpl;

template <>
struct TempImpl<via::Device::CPU> {
    static void execute(Tensor& a);
};

// 显式实例化声明
extern template struct TempImpl<via::Device::CPU>;
}