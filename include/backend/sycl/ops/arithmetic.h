#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

// *****************************************
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
// *****************************************
template <>
struct AddImpl<Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct SubImpl<Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};
template <>
struct DotImpl<Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};
template <>
struct DivImpl<Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};  
template <>
struct SinImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct CosImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct PowImpl<Device::SYCL> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct LogImpl<Device::SYCL> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct ExpImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SqrtImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct AbsImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct ClampImpl<Device::SYCL> {
    static void execute(Tensor& a,float min,float max);
    static Tensor execute(const Tensor& a,float min,float max);
};

extern template struct AddImpl<Device::SYCL>;
extern template struct SubImpl<Device::SYCL>;
extern template struct DotImpl<Device::SYCL>;
extern template struct DivImpl<Device::SYCL>;
extern template struct SinImpl<Device::SYCL>;
extern template struct CosImpl<Device::SYCL>;
extern template struct TanImpl<Device::SYCL>;
extern template struct ExpImpl<Device::SYCL>;
extern template struct SqrtImpl<Device::SYCL>;
extern template struct PowImpl<Device::SYCL>;
extern template struct LogImpl<Device::SYCL>;
extern template struct ClampImpl<Device::SYCL>;
extern template struct AbsImpl<Device::SYCL>;

}