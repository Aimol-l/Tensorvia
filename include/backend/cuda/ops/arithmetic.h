#pragma once
#include "backend/cuda/cuda_tensor.h"

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
struct AddImpl<Device::CUDA> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <>
struct SubImpl<Device::CUDA> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <>
struct DotImpl<Device::CUDA> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <>
struct DivImpl<Device::CUDA> {
    static void execute(Tensor& a,float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
};  
template <>
struct SinImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct CosImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct PowImpl<Device::CUDA> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct LogImpl<Device::CUDA> {
    static void execute(Tensor& a,float val);
    static Tensor execute(const Tensor& a,float val);
};
template <>
struct ExpImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SqrtImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct AbsImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct ClampImpl<Device::CUDA> {
    static void execute(Tensor& a,float min,float max);
    static Tensor execute(const Tensor& a,float min,float max);
};

extern template struct AddImpl<Device::CUDA>;
extern template struct SubImpl<Device::CUDA>;
extern template struct DotImpl<Device::CUDA>;
extern template struct DivImpl<Device::CUDA>;
extern template struct SinImpl<Device::CUDA>;
extern template struct CosImpl<Device::CUDA>;
extern template struct TanImpl<Device::CUDA>;
extern template struct PowImpl<Device::CUDA>;
extern template struct LogImpl<Device::CUDA>;
extern template struct ExpImpl<Device::CUDA>;
extern template struct SqrtImpl<Device::CUDA>;
extern template struct AbsImpl<Device::CUDA>;
extern template struct ClampImpl<Device::CUDA>;

}