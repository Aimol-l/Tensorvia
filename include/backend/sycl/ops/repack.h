#pragma once
#include "core/types.h"
#include "core/tensor.h"
#include "sycl_context.h"

template <via::Device D> struct RepackImpl;

template <>
struct RepackImpl<via::Device::SYCL> {
    static void execute(const Metadata& meta,void* input,void* output,std::shared_ptr<SYCLContext> ctx);
};

// 显式实例化声明
extern template struct RepackImpl<via::Device::SYCL>;