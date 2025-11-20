#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <Device D> struct SumImpl;
template <Device D> struct MeanImpl;
template <Device D> struct MinImpl;
template <Device D> struct MaxImpl;
template <Device D> struct ArgMaxImpl;
template <Device D> struct ArgMinImpl;
template <Device D> struct AllImpl;
template <Device D> struct AnyImpl;

template <>
struct SumImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MinImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MaxImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct MeanImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct ArgMaxImpl<Device::VULKAN> {
   static Tensor execute(const Tensor& a, int axis);
};

template <>
struct ArgMinImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct AnyImpl<Device::VULKAN> {
    static bool execute(const Tensor& a,float val);
};

template <>
struct AllImpl<Device::VULKAN> {
    static bool execute(const Tensor& a,float val);
};

extern template struct SumImpl<Device::VULKAN>;
extern template struct MinImpl<Device::VULKAN>;
extern template struct MaxImpl<Device::VULKAN>;   
extern template struct MeanImpl<Device::VULKAN>;
extern template struct ArgMaxImpl<Device::VULKAN>;
extern template struct ArgMinImpl<Device::VULKAN>;
extern template struct AnyImpl<Device::VULKAN>;
extern template struct AllImpl<Device::VULKAN>;

}