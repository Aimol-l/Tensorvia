#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

// ************************************** 声明 **************************************
template <via::Device D>
struct SumImpl;
template <via::Device D>
struct MeanImpl;
template <via::Device D>
struct MaxImpl;
template <via::Device D>
struct MinImpl;
template <via::Device D>
struct ArgMaxImpl;
template <via::Device D>
struct ArgMinImpl;
template <via::Device D>
struct AllImpl;
template <via::Device D>
struct AnyImpl;
// ************************************** 特化 **************************************
template <>
struct SumImpl<via::Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MinImpl<via::Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MaxImpl<via::Device::CPU> {
    static float execute(const Tensor& a);

    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct MeanImpl<via::Device::CPU> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct ArgMaxImpl<via::Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};
template <>
struct ArgMinImpl<via::Device::CPU> {
    static Tensor execute(const Tensor& a, int axis);
};

template <>
struct AllImpl<via::Device::CPU> {
    static bool execute(const Tensor& a, float value);
};

template <>
struct AnyImpl<via::Device::CPU> {
    static bool execute(const Tensor& a, float value);
};

extern template struct SumImpl<via::Device::CPU>;
extern template struct MeanImpl<via::Device::CPU>;
extern template struct MaxImpl<via::Device::CPU>;
extern template struct MinImpl<via::Device::CPU>;
extern template struct ArgMaxImpl<via::Device::CPU>;
extern template struct ArgMinImpl<via::Device::CPU>;
extern template struct AllImpl<via::Device::CPU>;
extern template struct AnyImpl<via::Device::CPU>;

}  // namespace ops