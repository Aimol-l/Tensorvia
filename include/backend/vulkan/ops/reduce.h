#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {

template <via::Device D> struct SumImpl;
template <via::Device D> struct MeanImpl;
template <via::Device D> struct MinImpl;
template <via::Device D> struct MaxImpl;
template <via::Device D> struct ArgMaxImpl;
template <via::Device D> struct ArgMinImpl;
template <via::Device D> struct AllImpl;
template <via::Device D> struct AnyImpl;

template <>
struct SumImpl<via::Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MinImpl<via::Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MaxImpl<via::Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct MeanImpl<via::Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct ArgMaxImpl<via::Device::VULKAN> {
   static Tensor execute(const Tensor& a, int axis);
};

template <>
struct ArgMinImpl<via::Device::VULKAN> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct AnyImpl<via::Device::VULKAN> {
    static bool execute(const Tensor& a,float val);
};

template <>
struct AllImpl<via::Device::VULKAN> {
    static bool execute(const Tensor& a,float val);
};

extern template struct SumImpl<via::Device::VULKAN>;
extern template struct MinImpl<via::Device::VULKAN>;
extern template struct MaxImpl<via::Device::VULKAN>;   
extern template struct MeanImpl<via::Device::VULKAN>;
extern template struct ArgMaxImpl<via::Device::VULKAN>;
extern template struct ArgMinImpl<via::Device::VULKAN>;
extern template struct AnyImpl<via::Device::VULKAN>;
extern template struct AllImpl<via::Device::VULKAN>;

}