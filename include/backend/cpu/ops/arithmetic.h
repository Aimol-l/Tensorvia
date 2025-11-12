#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <Device D>
struct AddImpl;
template <Device D>
struct SubImpl;
template <Device D>
struct DotImpl;
template <Device D>
struct DivImpl;
template <Device D>
struct SinImpl;
template <Device D>
struct CosImpl;
template <Device D>
struct TanImpl;
template <Device D>
struct PowImpl;
template <Device D>
struct LogImpl;
template <Device D>
struct ExpImpl;
template <Device D>
struct SqrtImpl;
template <Device D>
struct AbsImpl;
template <Device D>
struct ClampImpl;

template <>
struct AddImpl<Device::CPU> {
    // inplace
    static void execute(Tensor& a, float b);
    // uninplace
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);

    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};

template <>
struct SubImpl<Device::CPU> {
    static void execute(Tensor& a, float b);

    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
};

template <>
struct DotImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
};

template <>
struct DivImpl<Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
};

template <>
struct AbsImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SinImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct CosImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct TanImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct ExpImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SqrtImpl<Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct LogImpl<Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};

template <>
struct PowImpl<Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};

template <>
struct ClampImpl<Device::CPU> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

extern template struct AddImpl<Device::CPU>;
extern template struct SubImpl<Device::CPU>;
extern template struct DotImpl<Device::CPU>;
extern template struct DivImpl<Device::CPU>;
extern template struct SinImpl<Device::CPU>;
extern template struct CosImpl<Device::CPU>;
extern template struct TanImpl<Device::CPU>;
extern template struct PowImpl<Device::CPU>;
extern template struct LogImpl<Device::CPU>;
extern template struct ExpImpl<Device::CPU>;
extern template struct SqrtImpl<Device::CPU>;
extern template struct AbsImpl<Device::CPU>;
extern template struct ClampImpl<Device::CPU>;
}  // namespace ops