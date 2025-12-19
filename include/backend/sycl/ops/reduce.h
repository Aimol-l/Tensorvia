#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"
#include <sycl/sycl.hpp>
#include <limits>

namespace ops {

//**************************************************
template <Device D> struct SumImpl;
template <Device D> struct MeanImpl;
template <Device D> struct MinImpl;
template <Device D> struct MaxImpl;
template <Device D> struct ArgMaxImpl;
template <Device D> struct ArgMinImpl;
template <Device D> struct AllImpl;
template <Device D> struct AnyImpl;
//**************************************************
template <>
struct SumImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MeanImpl<Device::SYCL> {
    static float execute(const Tensor& a) ;
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MinImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct MaxImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};
template <>
struct ArgMaxImpl<Device::SYCL> {
    static Tensor execute(const Tensor &a, int axis);
};
template <>
struct ArgMinImpl<Device::SYCL> {
    static Tensor execute(const Tensor &a, int axis) ;
};

template <>
struct AnyImpl<Device::SYCL> {
    static bool execute(const Tensor& a,float val) ;
};
template <>
struct AllImpl<Device::SYCL> {
    static bool execute(const Tensor& a,float val) ;
};

extern template struct SumImpl<Device::SYCL>;
extern template struct MeanImpl<Device::SYCL>;
extern template struct MinImpl<Device::SYCL>;
extern template struct MaxImpl<Device::SYCL>;
extern template struct ArgMaxImpl<Device::SYCL>;
extern template struct ArgMinImpl<Device::SYCL>;
extern template struct AnyImpl<Device::SYCL>;
extern template struct AllImpl<Device::SYCL>;

}