#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

template <via::Device D>
struct AddImpl;
template <via::Device D>
struct SubImpl;
template <via::Device D>
struct DotImpl;
template <via::Device D>
struct DivImpl;
template <via::Device D>
struct SinImpl;
template <via::Device D>
struct CosImpl;
template <via::Device D>
struct TanImpl;
template <via::Device D>
struct PowImpl;
template <via::Device D>
struct LogImpl;
template <via::Device D>
struct ExpImpl;
template <via::Device D>
struct SqrtImpl;
template <via::Device D>
struct AbsImpl;
template <via::Device D>
struct ClampImpl;

template <>
struct AddImpl<via::Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};

template <>
struct SubImpl<via::Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);
};

template <>
struct DotImpl<via::Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};

template <>
struct DivImpl<via::Device::CPU> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float b);
    static void execute(const Tensor& a, const Tensor& b,Tensor& dst);

};

template <>
struct AbsImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SinImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct CosImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct TanImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct ExpImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct SqrtImpl<via::Device::CPU> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};

template <>
struct LogImpl<via::Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};

template <>
struct PowImpl<via::Device::CPU> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};

template <>
struct ClampImpl<via::Device::CPU> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

extern template struct AddImpl<via::Device::CPU>;
extern template struct SubImpl<via::Device::CPU>;
extern template struct DotImpl<via::Device::CPU>;
extern template struct DivImpl<via::Device::CPU>;
extern template struct SinImpl<via::Device::CPU>;
extern template struct CosImpl<via::Device::CPU>;
extern template struct TanImpl<via::Device::CPU>;
extern template struct PowImpl<via::Device::CPU>;
extern template struct LogImpl<via::Device::CPU>;
extern template struct ExpImpl<via::Device::CPU>;
extern template struct SqrtImpl<via::Device::CPU>;
extern template struct AbsImpl<via::Device::CPU>;
extern template struct ClampImpl<via::Device::CPU>;
}  // namespace ops