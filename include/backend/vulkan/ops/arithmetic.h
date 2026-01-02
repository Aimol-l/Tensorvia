#pragma once
#include "backend/vulkan/vulkan_tensor.h"

namespace ops {
template <via::Device D> struct AddImpl;
template <via::Device D> struct SubImpl;
template <via::Device D> struct DotImpl;
template <via::Device D> struct DivImpl;
template <via::Device D> struct SinImpl;
template <via::Device D> struct CosImpl;
template <via::Device D> struct TanImpl;
template <via::Device D> struct PowImpl;
template <via::Device D> struct LogImpl;
template <via::Device D> struct ExpImpl;
template <via::Device D> struct SqrtImpl;
template <via::Device D> struct AbsImpl;
template <via::Device D> struct ClampImpl;


template <>
struct AddImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct SubImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct DotImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct DivImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};  
template <>
struct SinImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct CosImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct PowImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct LogImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct ExpImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SqrtImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct AbsImpl<via::Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct ClampImpl<via::Device::VULKAN> {
    static void execute(Tensor& a,float min,float max);
    static Tensor execute(const Tensor& a,float min,float max);
};

extern template struct AddImpl<via::Device::VULKAN>;
extern template struct SubImpl<via::Device::VULKAN>;
extern template struct DotImpl<via::Device::VULKAN>;
extern template struct DivImpl<via::Device::VULKAN>;
extern template struct SinImpl<via::Device::VULKAN>;
extern template struct CosImpl<via::Device::VULKAN>;
extern template struct TanImpl<via::Device::VULKAN>;
extern template struct PowImpl<via::Device::VULKAN>;
extern template struct LogImpl<via::Device::VULKAN>;
extern template struct ExpImpl<via::Device::VULKAN>;
extern template struct SqrtImpl<via::Device::VULKAN>;
extern template struct AbsImpl<via::Device::VULKAN>;
extern template struct ClampImpl<via::Device::VULKAN>;

}