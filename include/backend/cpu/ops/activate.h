#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <via::Device D>
struct ReluImpl;

template <via::Device D>
struct SiluImpl;

template <via::Device D>
struct TanhImpl;

template <via::Device D>
struct SigmoidImpl;

template <via::Device D>
struct SoftmaxImpl;

template <>
struct ReluImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SiluImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a) ;
};

template <>
struct TanhImpl<via::Device::CPU> {
    static void execute(Tensor& a) ;
    static Tensor execute(const Tensor& a);
};
    
template <>
struct SigmoidImpl<via::Device::CPU> {
    static void execute(Tensor& a) ;
    static Tensor execute(const Tensor& a);
};
template <>
struct SoftmaxImpl<via::Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};

extern template struct ReluImpl<via::Device::CPU>;
extern template struct SiluImpl<via::Device::CPU>;
extern template struct TanhImpl<via::Device::CPU>;
extern template struct SigmoidImpl<via::Device::CPU>;
extern template struct SoftmaxImpl<via::Device::CPU>;

}