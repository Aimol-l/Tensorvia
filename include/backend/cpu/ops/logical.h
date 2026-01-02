#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

//***************************************************
template <via::Device D>
struct EqualImpl;

template <via::Device D>
struct NotEqualImpl;

template <via::Device D>
struct GreaterImpl;

template <via::Device D>
struct LessImpl;

template <via::Device D>
struct GreaterEqualImpl;

template <via::Device D>
struct LessEqualImpl;

template <via::Device D>
struct NonZeroImpl;
//***************************************************

template <>
struct EqualImpl<via::Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

template <>
struct LessImpl<via::Device::CPU> {
    static Tensor execute(const Tensor &a, const Tensor &b);
};

template <>
struct GreaterImpl<via::Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};
template <>
struct LessEqualImpl<via::Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};
template <>
struct GreaterEqualImpl<via::Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b);
};
template <>
struct NotEqualImpl<via::Device::CPU> {
     static Tensor execute(const Tensor &a, const Tensor &b) ;
};

template <>
struct NonZeroImpl<via::Device::CPU> {
    static size_t execute(const Tensor& a);
};

extern template struct EqualImpl<via::Device::CPU>;
extern template struct LessImpl<via::Device::CPU>;
extern template struct GreaterImpl<via::Device::CPU>;
extern template struct LessEqualImpl<via::Device::CPU>;
extern template struct GreaterEqualImpl<via::Device::CPU>;
extern template struct NotEqualImpl<via::Device::CPU>;
extern template struct NonZeroImpl<via::Device::CPU>;

}