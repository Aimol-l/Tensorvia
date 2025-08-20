#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


//************************************
template <Device D>
struct EqualImpl;

template <Device D>
struct NotEqualImpl;

template <Device D>
struct GreaterImpl;

template <Device D>
struct LessImpl;

template <Device D>
struct GreaterEqualImpl;

template <Device D>
struct LessEqualImpl;

template <Device D>
struct NonZeroImpl;
//************************************
template <>
struct EqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct NotEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;

};
template <>
struct GreaterImpl<Device::SYCL> {
   static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct LessImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct GreaterEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct LessEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct NonZeroImpl<Device::SYCL> {
    static size_t execute(const Tensor& a) ;
};

extern template struct EqualImpl<Device::SYCL>;
extern template struct NotEqualImpl<Device::SYCL>;
extern template struct GreaterImpl<Device::SYCL>;
extern template struct LessImpl<Device::SYCL>;
extern template struct GreaterEqualImpl<Device::SYCL>;
extern template struct LessEqualImpl<Device::SYCL>;
extern template struct NonZeroImpl<Device::SYCL>;


}