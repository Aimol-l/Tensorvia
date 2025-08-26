#pragma once
#include "backend/cuda/cuda_tensor.h"

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
struct SumImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MinImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MaxImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct MeanImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct ArgMaxImpl<Device::CUDA> {
   static Tensor execute(const Tensor& a, int axis);
};

template <>
struct ArgMinImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct AnyImpl<Device::CUDA> {
    static bool execute(const Tensor& a,float val);
};

template <>
struct AllImpl<Device::CUDA> {
    static bool execute(const Tensor& a,float val);
};

extern template struct SumImpl<Device::CUDA>;
extern template struct MinImpl<Device::CUDA>;
extern template struct MaxImpl<Device::CUDA>;   
extern template struct MeanImpl<Device::CUDA>;
extern template struct ArgMaxImpl<Device::CUDA>;
extern template struct ArgMinImpl<Device::CUDA>;
extern template struct AnyImpl<Device::CUDA>;
extern template struct AllImpl<Device::CUDA>;

}