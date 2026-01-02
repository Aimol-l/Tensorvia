#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <via::Device D> struct EqualImpl;
template <via::Device D> struct NotEqualImpl;
template <via::Device D> struct GreaterImpl;
template <via::Device D> struct LessImpl;
template <via::Device D> struct GreaterEqualImpl;
template <via::Device D> struct LessEqualImpl;
template <via::Device D> struct NonZeroImpl;

template <>
struct EqualImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NotEqualImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterEqualImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessEqualImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NonZeroImpl<via::Device::CUDA> {
    static size_t execute(const Tensor& a);
};

extern template struct EqualImpl<via::Device::CUDA>;
extern template struct NotEqualImpl<via::Device::CUDA>;
extern template struct GreaterImpl<via::Device::CUDA>;
extern template struct LessImpl<via::Device::CUDA>;
extern template struct GreaterEqualImpl<via::Device::CUDA>;
extern template struct LessEqualImpl<via::Device::CUDA>;
extern template struct NonZeroImpl<via::Device::CUDA>;
}