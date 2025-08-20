#pragma once
#include "backend/cuda/cuda_tensor.h"

namespace ops { 

template <Device D> struct ZerosImpl;
template <Device D> struct OnesImpl;
template <Device D> struct FillImpl;
template <Device D> struct RandomImpl;

template <>
struct ZerosImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int>& shape, DataType dtype);
};

template <>
struct OnesImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int>& shape, DataType dtype);
};

template <>
struct FillImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int>& shape, DataType dtype, float value);
};

template <>
struct RandomImpl<Device::CUDA> { 
    static Tensor execute(const std::vector<int>& shape, DataType dtype,float min,float max);
};

extern template struct ZerosImpl<Device::CUDA>;
extern template struct OnesImpl<Device::CUDA>;
extern template struct FillImpl<Device::CUDA>;
extern template struct RandomImpl<Device::CUDA>;

}