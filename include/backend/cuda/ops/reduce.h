#pragma once
#include "backend/cuda/cuda_tensor.h"

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
struct SumImpl<via::Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MinImpl<via::Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct MaxImpl<via::Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct MeanImpl<via::Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a,int axis);
};

template <>
struct ArgMaxImpl<via::Device::CUDA> {
   static Tensor execute(const Tensor& a, int axis);
};

template <>
struct ArgMinImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct AnyImpl<via::Device::CUDA> {
    static bool execute(const Tensor& a,float val);
};

template <>
struct AllImpl<via::Device::CUDA> {
    static bool execute(const Tensor& a,float val);
};

extern template struct SumImpl<via::Device::CUDA>;
extern template struct MinImpl<via::Device::CUDA>;
extern template struct MaxImpl<via::Device::CUDA>;   
extern template struct MeanImpl<via::Device::CUDA>;
extern template struct ArgMaxImpl<via::Device::CUDA>;
extern template struct ArgMinImpl<via::Device::CUDA>;
extern template struct AnyImpl<via::Device::CUDA>;
extern template struct AllImpl<via::Device::CUDA>;

}