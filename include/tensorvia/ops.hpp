#pragma once

#include <cstddef>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

namespace ops {

template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void println(Tensor& a);
void println(Tensor&& a);

void export_csv(Tensor& a, const std::string& path);
void export_csv(Tensor&& a, const std::string& path);

Tensor Ones(const std::vector<int64_t>& shape, DataType dtype, Device device = Device::CPU);
Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, Device device = Device::CPU);
Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float value, Device device = Device::CPU);
Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, Device device = Device::CPU);
Tensor Randn(const std::vector<int64_t>& shape, DataType dtype, float mean = 0.0f, float std = 1.0f,
             Device device = Device::CPU);

Tensor Arange(float start, float stop, float step = 1.0f, DataType dtype = DataType::FLOAT32);
Tensor Linspace(float start, float stop, int64_t steps, DataType dtype = DataType::FLOAT32);
Tensor Eye(int64_t n, DataType dtype = DataType::FLOAT32);
Tensor Eye(int64_t n, int64_t m, DataType dtype = DataType::FLOAT32);

void Zeros(Tensor& a);
void Ones(Tensor& a);
void Fill(Tensor& a, float value);

Tensor Slice(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);

void Add(Tensor& a, float b);
void Sub(Tensor& a, float b);
void Mul(Tensor& a, float b);
void Div(Tensor& a, float b);

Tensor Add(const Tensor& a, float b);
Tensor Sub(const Tensor& a, float b);
Tensor Mul(const Tensor& a, float b);
Tensor Div(const Tensor& a, float b);

Tensor Add(const Tensor& a, const Tensor& b);
Tensor Sub(const Tensor& a, const Tensor& b);
Tensor Mul(const Tensor& a, const Tensor& b);
Tensor Div(const Tensor& a, const Tensor& b);

void Add(const Tensor& a, const Tensor& b, Tensor& dst);
void Sub(const Tensor& a, const Tensor& b, Tensor& dst);
void Mul(const Tensor& a, const Tensor& b, Tensor& dst);
void Div(const Tensor& a, const Tensor& b, Tensor& dst);

Tensor MatMul(const Tensor& a, const Tensor& b);

void Abs(Tensor& a);
void Relu(Tensor& a);
void Sin(Tensor& a);
void Cos(Tensor& a);
void Tan(Tensor& a);
void Silu(Tensor& a);
void Tanh(Tensor& a);
void Sigmoid(Tensor& a);
void Pow(Tensor& a, float val);
void Log(Tensor& a, float val);
void Clamp(Tensor& a, float min, float max);

Tensor Abs(const Tensor& a);
Tensor Sin(const Tensor& a);
Tensor Cos(const Tensor& a);
Tensor Tan(const Tensor& a);
Tensor Exp(const Tensor& a);
Tensor Relu(const Tensor& a);
Tensor Silu(const Tensor& a);
Tensor Tanh(const Tensor& a);
Tensor Sqrt(const Tensor& a);
Tensor Sigmoid(const Tensor& a);
Tensor Pow(const Tensor& a, float val);
Tensor Log(const Tensor& a, float val);
Tensor Softmax(const Tensor& a, int axis);
Tensor LogSoftmax(const Tensor& a, int axis);
Tensor Erf(const Tensor& a);
Tensor Clamp(const Tensor& a, float min, float max);

Tensor LayerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps = 1e-5f);
Tensor RmsNorm(const Tensor& a, const Tensor& weight, int axis, float eps = 1e-5f);

float Sum(const Tensor& a);
float Min(const Tensor& a);
float Max(const Tensor& a);
float Mean(const Tensor& a);
float Prod(const Tensor& a);
float Var(const Tensor& a);
float Std(const Tensor& a);
float Norm(const Tensor& a, float p = 2.0f);

Tensor Sum(const Tensor& a, int axis);
Tensor Min(const Tensor& a, int axis);
Tensor Max(const Tensor& a, int axis);
Tensor Mean(const Tensor& a, int axis);
Tensor Prod(const Tensor& a, int axis);
Tensor Var(const Tensor& a, int axis);
Tensor Std(const Tensor& a, int axis);
Tensor Norm(const Tensor& a, float p, int axis);
Tensor Cumsum(const Tensor& a, int axis);

float MseLoss(const Tensor& input, const Tensor& target);
float NllLoss(const Tensor& input, const Tensor& target);
float CrossEntropyLoss(const Tensor& input, const Tensor& target);

void Typecast(Tensor& a, DataType dst_type);
Tensor Typecast(const Tensor& a, DataType dst_type);
Tensor Concat(const std::vector<Tensor>& tensors, int dim);

void Transpose(Tensor& a);
Tensor Transpose(const Tensor& a, std::initializer_list<int64_t> axes);
void Transpose(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);

Tensor Equal(const Tensor& a, const Tensor& b);
Tensor NotEqual(const Tensor& a, const Tensor& b);
Tensor Greater(const Tensor& a, const Tensor& b);
Tensor Less(const Tensor& a, const Tensor& b);
Tensor GreaterEqual(const Tensor& a, const Tensor& b);
Tensor LessEqual(const Tensor& a, const Tensor& b);

Tensor Equal(const Tensor& a, float val);
Tensor NotEqual(const Tensor& a, float val);
Tensor Greater(const Tensor& a, float val);
Tensor Less(const Tensor& a, float val);
Tensor GreaterEqual(const Tensor& a, float val);
Tensor LessEqual(const Tensor& a, float val);

Tensor Where(const Tensor& cond, const Tensor& a, const Tensor& b);
Tensor Where(const Tensor& cond, const Tensor& a, float b);
Tensor Where(const Tensor& cond, float a, const Tensor& b);

bool All(const Tensor& t, float val);
bool Any(const Tensor& t, float val);
size_t Nonzero(const Tensor& t);
Tensor Argmax(const Tensor& t, int axis);
Tensor Argmin(const Tensor& t, int axis);
Tensor IndexSelect(const Tensor& input, int dim, const Tensor& index);
Tensor Gather(const Tensor& input, int dim, const Tensor& index);

void Temp(Tensor& t);

}  // namespace ops
