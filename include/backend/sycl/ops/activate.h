
#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


template <via::Device D> struct ReluImpl;
template <via::Device D> struct SiluImpl;
template <via::Device D> struct TanhImpl;
template <via::Device D> struct SigmoidImpl;
template <via::Device D> struct SoftmaxImpl;

template <>
struct ReluImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SiluImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct TanhImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SigmoidImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SoftmaxImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,int axis);
};

extern template struct ReluImpl<via::Device::SYCL>;
extern template struct SiluImpl<via::Device::SYCL>;
extern template struct TanhImpl<via::Device::SYCL>;
extern template struct SigmoidImpl<via::Device::SYCL>;
extern template struct SoftmaxImpl<via::Device::SYCL>;

}