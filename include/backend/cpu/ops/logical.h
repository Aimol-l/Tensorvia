#pragma once
#include "backend/cpu/cpu_tensor.h"
#include <execution>
#include "ops.h"

namespace ops {

//***************************************************
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
//***************************************************

template <>
struct EqualImpl<Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

template <>
struct LessImpl<Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

template <>
struct GreaterImpl<Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};
template <>
struct LessEqualImpl<Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};
template <>
struct GreaterEqualImpl<Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b);
};
template <>
struct NotEqualImpl<Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};

template <>
struct NonZeroImpl<Device::CPU> {
    static size_t execute(const Tensor& a);
};

extern template struct EqualImpl<Device::CPU>;
extern template struct LessImpl<Device::CPU>;
extern template struct GreaterImpl<Device::CPU>;
extern template struct LessEqualImpl<Device::CPU>;
extern template struct GreaterEqualImpl<Device::CPU>;
extern template struct NotEqualImpl<Device::CPU>;
extern template struct NonZeroImpl<Device::CPU>;

}