#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

// ************************************** 声明 **************************************
template <Device D>
struct SumImpl;
template <Device D>
struct MeanImpl;
template <Device D>
struct MaxImpl;
template <Device D>
struct MinImpl;
template <Device D>
struct ArgMaxImpl;
template <Device D>
struct ArgMinImpl;
template <Device D>
struct AllImpl;
template <Device D>
struct AnyImpl;
// ************************************** 特化 **************************************
template <>
struct SumImpl<Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MinImpl<Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MaxImpl<Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MeanImpl<Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct ArgMaxImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct ArgMinImpl<Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct AllImpl<Device::CPU> {
    static bool execute(const Tensor& a, float value);
};

template <>
struct AnyImpl<Device::CPU> {
    static bool execute(const Tensor& a, float value);
};

extern template struct SumImpl<Device::CPU>;
extern template struct MeanImpl<Device::CPU>;
extern template struct MaxImpl<Device::CPU>;
extern template struct MinImpl<Device::CPU>;
extern template struct ArgMaxImpl<Device::CPU>;
extern template struct ArgMinImpl<Device::CPU>;
extern template struct AllImpl<Device::CPU>;
extern template struct AnyImpl<Device::CPU>;

}  // namespace ops