#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

// 前向声明模板
template <Device D> struct ReluImpl;
template <Device D> struct SiluImpl;
template <Device D> struct TanhImpl;
template <Device D> struct SigmoidImpl;
template <Device D> struct SoftmaxImpl;

// 2. 特化声明（但不包含实现）
template <>
struct ReluImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SiluImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanhImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SigmoidImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SoftmaxImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};

// 显式实例化声明
extern template struct ReluImpl<Device::CUDA>;
extern template struct SiluImpl<Device::CUDA>;
extern template struct TanhImpl<Device::CUDA>;
extern template struct SigmoidImpl<Device::CUDA>;
extern template struct SoftmaxImpl<Device::CUDA>;


// template <>
// struct ReluImpl<Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };
// template <>
// struct SiluImpl<Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };

// template <>
// struct TanhImpl<Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };
// template <>
// struct SigmoidImpl<Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };

// template <>
// struct SoftmaxImpl<Device::CUDA> {
//     static Tensor execute(const Tensor& a,int axis);
// };

}