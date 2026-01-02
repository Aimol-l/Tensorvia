#pragma once
#include "backend/cpu/cpu_tensor.h"


namespace ops { 

//****************************** 声明 ***********************************
template <via::Device D>
struct ZerosImpl;

template <via::Device D>
struct OnesImpl;

template <via::Device D>
struct FillImpl;

template <via::Device D>
struct RandomImpl;

//****************************** 特化 ***********************************

template <>
struct ZerosImpl<via::Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct OnesImpl<via::Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct FillImpl<via::Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float value);
};

template <>
struct RandomImpl<via::Device::CPU> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float min, float max);
};

extern template struct ZerosImpl<via::Device::CPU>;
extern template struct OnesImpl<via::Device::CPU>;
extern template struct FillImpl<via::Device::CPU>;
extern template struct RandomImpl<via::Device::CPU>;
}