#pragma once
#include "core/context.h"
#include "core/types.h"
#include "core/tensor.h"


template <via::Device D> struct RepackImpl;

template <>
struct RepackImpl<via::Device::CUDA> {
    static void execute(const Metadata& meta,void* input,void* output);
};

// 显式实例化声明
extern template struct RepackImpl<via::Device::CUDA>;