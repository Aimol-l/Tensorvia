#pragma once
#include "backend/cpu/cpu_tensor.h"


namespace ops { 

//****************************** 声明 ***********************************
template <Device D>
struct ZerosImpl;

template <Device D>
struct OnesImpl;

template <Device D>
struct FillImpl;

template <Device D>
struct RandomImpl;

//****************************** 特化 ***********************************

template <>
struct ZerosImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct OnesImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};

template <>
struct FillImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
};

template <>
struct RandomImpl<Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max);
};

extern template struct ZerosImpl<Device::CPU>;
extern template struct OnesImpl<Device::CPU>;
extern template struct FillImpl<Device::CPU>;
extern template struct RandomImpl<Device::CPU>;
}