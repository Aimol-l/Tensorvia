#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <Device D> struct EqualImpl;
template <Device D> struct NotEqualImpl;
template <Device D> struct GreaterImpl;
template <Device D> struct LessImpl;
template <Device D> struct GreaterEqualImpl;
template <Device D> struct LessEqualImpl;
template <Device D> struct NonZeroImpl;

template <>
struct EqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NotEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct GreaterEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct LessEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a,const Tensor& b);
};
template <>
struct NonZeroImpl<Device::VULKAN> {
    static size_t execute(const Tensor& a);
};

extern template struct EqualImpl<Device::VULKAN>;
extern template struct NotEqualImpl<Device::VULKAN>;
extern template struct GreaterImpl<Device::VULKAN>;
extern template struct LessImpl<Device::VULKAN>;
extern template struct GreaterEqualImpl<Device::VULKAN>;
extern template struct LessEqualImpl<Device::VULKAN>;
extern template struct NonZeroImpl<Device::VULKAN>;
}