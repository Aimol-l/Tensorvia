#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"
#include <sycl/sycl.hpp>
#include <limits>

namespace ops {

//**************************************************
template <via::Device D> struct SumImpl;
template <via::Device D> struct MeanImpl;
template <via::Device D> struct MinImpl;
template <via::Device D> struct MaxImpl;
template <via::Device D> struct ArgMaxImpl;
template <via::Device D> struct ArgMinImpl;
template <via::Device D> struct AllImpl;
template <via::Device D> struct AnyImpl;
//**************************************************
template <>
struct SumImpl<via::Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MeanImpl<via::Device::SYCL> {
    static float execute(const Tensor& a) ;
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MinImpl<via::Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MaxImpl<via::Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct ArgMaxImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor &a, int axis);
};
template <>
struct ArgMinImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor &a, int axis) ;
};

template <>
struct AnyImpl<via::Device::SYCL> {
    static bool execute(const Tensor& a,float val) ;
};
template <>
struct AllImpl<via::Device::SYCL> {
    static bool execute(const Tensor& a,float val) ;
};

extern template struct SumImpl<via::Device::SYCL>;
extern template struct MeanImpl<via::Device::SYCL>;
extern template struct MinImpl<via::Device::SYCL>;
extern template struct MaxImpl<via::Device::SYCL>;
extern template struct ArgMaxImpl<via::Device::SYCL>;
extern template struct ArgMinImpl<via::Device::SYCL>;
extern template struct AnyImpl<via::Device::SYCL>;
extern template struct AllImpl<via::Device::SYCL>;

}