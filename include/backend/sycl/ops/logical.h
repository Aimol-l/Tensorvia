#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {


//************************************
template <via::Device D> struct EqualImpl;
template <via::Device D> struct NotEqualImpl;
template <via::Device D> struct GreaterImpl;
template <via::Device D> struct LessImpl;
template <via::Device D> struct GreaterEqualImpl;
template <via::Device D> struct LessEqualImpl;
template <via::Device D> struct NonZeroImpl;
//************************************

template <>
struct EqualImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct NotEqualImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;

};
template <>
struct GreaterImpl<via::Device::SYCL> {
   static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct LessImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct GreaterEqualImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct LessEqualImpl<via::Device::SYCL> {
    static Tensor execute(const Tensor& a,const Tensor& b) ;
};
template <>
struct NonZeroImpl<via::Device::SYCL> {
    static size_t execute(const Tensor& a) ;
};

extern template struct EqualImpl<via::Device::SYCL>;
extern template struct NotEqualImpl<via::Device::SYCL>;
extern template struct GreaterImpl<via::Device::SYCL>;
extern template struct LessImpl<via::Device::SYCL>;
extern template struct GreaterEqualImpl<via::Device::SYCL>;
extern template struct LessEqualImpl<via::Device::SYCL>;
extern template struct NonZeroImpl<via::Device::SYCL>;


}