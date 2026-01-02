#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops {

// 前向声明模板
template <via::Device D> struct ReluImpl;
template <via::Device D> struct SiluImpl;
template <via::Device D> struct TanhImpl;
template <via::Device D> struct SigmoidImpl;
template <via::Device D> struct SoftmaxImpl;

// 2. 特化声明（但不包含实现）
template <>
struct ReluImpl<via::Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SiluImpl<via::Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct TanhImpl<via::Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SigmoidImpl<via::Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <>
struct SoftmaxImpl<via::Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};

// 显式实例化声明
extern template struct ReluImpl<via::Device::CUDA>;
extern template struct SiluImpl<via::Device::CUDA>;
extern template struct TanhImpl<via::Device::CUDA>;
extern template struct SigmoidImpl<via::Device::CUDA>;
extern template struct SoftmaxImpl<via::Device::CUDA>;


// template <>
// struct ReluImpl<via::Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };
// template <>
// struct SiluImpl<via::Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };

// template <>
// struct TanhImpl<via::Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };
// template <>
// struct SigmoidImpl<via::Device::CUDA> {
//     static void execute(Tensor& a);
//     static Tensor execute(const Tensor& a);
// };

// template <>
// struct SoftmaxImpl<via::Device::CUDA> {
//     static Tensor execute(const Tensor& a,int axis);
// };

}