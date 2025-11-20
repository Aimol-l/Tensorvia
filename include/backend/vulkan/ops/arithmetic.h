#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {
template <Device D> struct AddImpl;
template <Device D> struct SubImpl;
template <Device D> struct DotImpl;
template <Device D> struct DivImpl;
template <Device D> struct SinImpl;
template <Device D> struct CosImpl;
template <Device D> struct TanImpl;
template <Device D> struct PowImpl;
template <Device D> struct LogImpl;
template <Device D> struct ExpImpl;
template <Device D> struct SqrtImpl;
template <Device D> struct AbsImpl;
template <Device D> struct ClampImpl;


template <>
struct AddImpl<Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct SubImpl<Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <>
struct DotImpl<Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <>
struct DivImpl<Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};  
template <>
struct SinImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct CosImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct PowImpl<Device::VULKAN> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct LogImpl<Device::VULKAN> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct ExpImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SqrtImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct AbsImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct ClampImpl<Device::VULKAN> {
    static void execute(Tensor& a,float min,float max);
    static Tensor execute(const Tensor& a,float min,float max);
};

extern template struct AddImpl<Device::VULKAN>;
extern template struct SubImpl<Device::VULKAN>;
extern template struct DotImpl<Device::VULKAN>;
extern template struct DivImpl<Device::VULKAN>;
extern template struct SinImpl<Device::VULKAN>;
extern template struct CosImpl<Device::VULKAN>;
extern template struct TanImpl<Device::VULKAN>;
extern template struct PowImpl<Device::VULKAN>;
extern template struct LogImpl<Device::VULKAN>;
extern template struct ExpImpl<Device::VULKAN>;
extern template struct SqrtImpl<Device::VULKAN>;
extern template struct AbsImpl<Device::VULKAN>;
extern template struct ClampImpl<Device::VULKAN>;

}