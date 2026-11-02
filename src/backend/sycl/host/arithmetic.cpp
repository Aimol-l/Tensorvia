// 为 Device::SYCL 定义 AddImpl/SubImpl/MulImpl/DivImpl/AbsImpl/... 的宿主驱动。
//
// 设备侧每次启动都与 dtype 无关：宿主在调用前把两个操作数（以及输出）提升到公共
// dtype，所以每次接缝调用只派发一种标量类型。混合 dtype 会被拒绝，而不是静默
// 收窄。
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace ops {

// ---------------------------------------------------------------------------
// 标量算术。
// ---------------------------------------------------------------------------
void AddImpl<Device::SYCL>::execute(Tensor& a, float b) {
    sycl_ufunc::scalar_inplace(a, tvv::sycl::kBinAdd, b, "Add");
}
void SubImpl<Device::SYCL>::execute(Tensor& a, float b) {
    sycl_ufunc::scalar_inplace(a, tvv::sycl::kBinSub, b, "Sub");
}
void MulImpl<Device::SYCL>::execute(Tensor& a, float b) {
    sycl_ufunc::scalar_inplace(a, tvv::sycl::kBinMul, b, "Mul");
}
void DivImpl<Device::SYCL>::execute(Tensor& a, float b) {
    sycl_ufunc::scalar_inplace(a, tvv::sycl::kBinDiv, b, "Div");
}

Tensor AddImpl<Device::SYCL>::execute(const Tensor& a, float b) {
    return sycl_ufunc::scalar_out(a, tvv::sycl::kBinAdd, b, "Add");
}
Tensor SubImpl<Device::SYCL>::execute(const Tensor& a, float b) {
    return sycl_ufunc::scalar_out(a, tvv::sycl::kBinSub, b, "Sub");
}
Tensor MulImpl<Device::SYCL>::execute(const Tensor& a, float b) {
    return sycl_ufunc::scalar_out(a, tvv::sycl::kBinMul, b, "Mul");
}
Tensor DivImpl<Device::SYCL>::execute(const Tensor& a, float b) {
    return sycl_ufunc::scalar_out(a, tvv::sycl::kBinDiv, b, "Div");
}

// ---------------------------------------------------------------------------
// 张量-张量算术：dtype 提升 + NumPy 广播。
// ---------------------------------------------------------------------------
Tensor AddImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::binary_arith(tvv::sycl::kBinAdd, a, b, nullptr);
}
Tensor SubImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::binary_arith(tvv::sycl::kBinSub, a, b, nullptr);
}
Tensor MulImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::binary_arith(tvv::sycl::kBinMul, a, b, nullptr);
}
Tensor DivImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::binary_arith(tvv::sycl::kBinDiv, a, b, nullptr);
}

void AddImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    sycl_ufunc::binary_arith(tvv::sycl::kBinAdd, a, b, &dst);
}
void SubImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    sycl_ufunc::binary_arith(tvv::sycl::kBinSub, a, b, &dst);
}
void MulImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    sycl_ufunc::binary_arith(tvv::sycl::kBinMul, a, b, &dst);
}
void DivImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    sycl_ufunc::binary_arith(tvv::sycl::kBinDiv, a, b, &dst);
}

// ---------------------------------------------------------------------------
// 一元逐元素运算。超越算子的整型输入以 FLOAT32 计算；Abs/Clamp 保持输入
// dtype。
// ---------------------------------------------------------------------------
namespace {

void unary_identity_inplace(Tensor& a, int op, float p0, float p1, const char* what) {
    sycl_ufunc::unary_inplace(a, op, p0, p1, what);
}
Tensor unary_identity_out(const Tensor& a, int op, float p0, float p1, const char* what) {
    return sycl_ufunc::unary_out(a, a.dtype(), op, p0, p1, what);
}
void unary_float_inplace(Tensor& a, int op, float p0, float p1, const char* what) {
    sycl_ufunc::unary_inplace(a, op, p0, p1, what);
}
Tensor unary_float_out(const Tensor& a, int op, float p0, float p1, const char* what) {
    return sycl_ufunc::unary_out(a, sycl_ufunc::float_result(a.dtype()), op, p0, p1, what);
}

}  // namespace

void AbsImpl<Device::SYCL>::execute(Tensor& a) {
    unary_identity_inplace(a, tvv::sycl::kUnaryAbs, 0.0f, 0.0f, "Abs");
}
Tensor AbsImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_identity_out(a, tvv::sycl::kUnaryAbs, 0.0f, 0.0f, "Abs");
}

void SinImpl<Device::SYCL>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::sycl::kUnarySin, 0.0f, 0.0f, "Sin");
}
Tensor SinImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::sycl::kUnarySin, 0.0f, 0.0f, "Sin");
}
void CosImpl<Device::SYCL>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::sycl::kUnaryCos, 0.0f, 0.0f, "Cos");
}
Tensor CosImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::sycl::kUnaryCos, 0.0f, 0.0f, "Cos");
}
void TanImpl<Device::SYCL>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::sycl::kUnaryTan, 0.0f, 0.0f, "Tan");
}
Tensor TanImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::sycl::kUnaryTan, 0.0f, 0.0f, "Tan");
}
void ExpImpl<Device::SYCL>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::sycl::kUnaryExp, 0.0f, 0.0f, "Exp");
}
Tensor ExpImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::sycl::kUnaryExp, 0.0f, 0.0f, "Exp");
}

Tensor ErfImpl<Device::SYCL>::execute(const Tensor& a) {
    // Erf 与 Exp 同族：整型输入在 double 里算（unary_compute_t）再提升到 FLOAT32。
    return unary_float_out(a, tvv::sycl::kUnaryErf, 0.0f, 0.0f, "Erf");
}
void SqrtImpl<Device::SYCL>::execute(Tensor& a) {
    unary_float_inplace(a, tvv::sycl::kUnarySqrt, 0.0f, 0.0f, "Sqrt");
}
Tensor SqrtImpl<Device::SYCL>::execute(const Tensor& a) {
    return unary_float_out(a, tvv::sycl::kUnarySqrt, 0.0f, 0.0f, "Sqrt");
}

void PowImpl<Device::SYCL>::execute(Tensor& a, float val) {
    unary_float_inplace(a, tvv::sycl::kUnaryPow, val, 0.0f, "Pow");
}
Tensor PowImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return unary_float_out(a, tvv::sycl::kUnaryPow, val, 0.0f, "Pow");
}
void LogImpl<Device::SYCL>::execute(Tensor& a, float val) {
    unary_float_inplace(a, tvv::sycl::kUnaryLog, val, 0.0f, "Log");
}
Tensor LogImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return unary_float_out(a, tvv::sycl::kUnaryLog, val, 0.0f, "Log");
}

void ClampImpl<Device::SYCL>::execute(Tensor& a, float min, float max) {
    unary_identity_inplace(a, tvv::sycl::kUnaryClamp, min, max, "Clamp");
}
Tensor ClampImpl<Device::SYCL>::execute(const Tensor& a, float min, float max) {
    return unary_identity_out(a, tvv::sycl::kUnaryClamp, min, max, "Clamp");
}

}  // namespace ops
