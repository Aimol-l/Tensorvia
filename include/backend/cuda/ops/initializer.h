#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops { 

template <via::Device D> struct ZerosImpl;
template <via::Device D> struct OnesImpl;
template <via::Device D> struct FillImpl;
template <via::Device D> struct RandomImpl;

template <>
struct ZerosImpl<via::Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct OnesImpl<via::Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype);
};

template <>
struct FillImpl<via::Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype, float value);
};

template <>
struct RandomImpl<via::Device::CUDA> { 
    static Tensor execute(const std::vector<int64_t>& shape, via::DataType dtype,float min,float max);
};

extern template struct ZerosImpl<via::Device::CUDA>;
extern template struct OnesImpl<via::Device::CUDA>;
extern template struct FillImpl<via::Device::CUDA>;
extern template struct RandomImpl<via::Device::CUDA>;

}