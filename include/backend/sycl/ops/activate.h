
#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


template <Device D>
struct ReluImpl;

template <Device D>
struct SiluImpl;

template <Device D>
struct TanhImpl;

template <Device D>
struct SigmoidImpl;

template <Device D>
struct SoftmaxImpl;
//****************************************************
template <>
struct ReluImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SiluImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct TanhImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SigmoidImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SoftmaxImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,int axis);
};

extern template struct ReluImpl<Device::SYCL>;
extern template struct SiluImpl<Device::SYCL>;
extern template struct TanhImpl<Device::SYCL>;
extern template struct SigmoidImpl<Device::SYCL>;
extern template struct SoftmaxImpl<Device::SYCL>;

}