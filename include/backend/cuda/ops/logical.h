#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

template <Device D> struct EqualImpl;
template <Device D> struct NotEqualImpl;
template <Device D> struct GreaterImpl;
template <Device D> struct LessImpl;
template <Device D> struct GreaterEqualImpl;
template <Device D> struct LessEqualImpl;
template <Device D> struct NonZeroImpl;

template <>
struct EqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NotEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NonZeroImpl<Device::CUDA> {
    static size_t execute(const Tensor& a);
};

extern template struct EqualImpl<Device::CUDA>;
extern template struct NotEqualImpl<Device::CUDA>;
extern template struct GreaterImpl<Device::CUDA>;
extern template struct LessImpl<Device::CUDA>;
extern template struct GreaterEqualImpl<Device::CUDA>;
extern template struct LessEqualImpl<Device::CUDA>;
extern template struct NonZeroImpl<Device::CUDA>;
}