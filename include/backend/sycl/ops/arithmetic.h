#pragma once
#include "ops.h"
#include "backend/sycl/sycl_tensor.h"

namespace ops {

// *****************************************
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
// *****************************************
template <>
struct AddImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};
template <>
struct SubImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};
template <>
struct DotImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};
template <>
struct DivImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};  
template <>
struct SinImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct CosImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct PowImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct LogImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct ExpImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SqrtImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct AbsImpl<via::Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct ClampImpl<via::Device::SYCL> {
    static void execute(Tensor& a,float min,float max);
    static Tensor execute(const Tensor& a,float min,float max);
};

extern template struct AddImpl<via::Device::SYCL>;
extern template struct SubImpl<via::Device::SYCL>;
extern template struct DotImpl<via::Device::SYCL>;
extern template struct DivImpl<via::Device::SYCL>;
extern template struct SinImpl<via::Device::SYCL>;
extern template struct CosImpl<via::Device::SYCL>;
extern template struct TanImpl<via::Device::SYCL>;
extern template struct ExpImpl<via::Device::SYCL>;
extern template struct SqrtImpl<via::Device::SYCL>;
extern template struct PowImpl<via::Device::SYCL>;
extern template struct LogImpl<via::Device::SYCL>;
extern template struct ClampImpl<via::Device::SYCL>;
extern template struct AbsImpl<via::Device::SYCL>;

}