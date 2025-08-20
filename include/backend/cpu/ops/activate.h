#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

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

template <>
struct ReluImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SiluImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a) ;
};

template <>
struct TanhImpl<Device::CPU> {
    static void execute(Tensor& a) ;
    static Tensor execute(const Tensor& a);
};
    
template <>
struct SigmoidImpl<Device::CPU> {
    static void execute(Tensor& a) ;
    static Tensor execute(const Tensor& a);
};
template <>
struct SoftmaxImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};

extern template struct ReluImpl<Device::CPU>;
extern template struct SiluImpl<Device::CPU>;
extern template struct TanhImpl<Device::CPU>;
extern template struct SigmoidImpl<Device::CPU>;
extern template struct SoftmaxImpl<Device::CPU>;

}