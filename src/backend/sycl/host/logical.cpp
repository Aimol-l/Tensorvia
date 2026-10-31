// Device::SYCL 的 Equal/NotEqual/Greater/Less/GreaterEqual/LessEqual。
//
// 宿主把两个操作数提升到公共 dtype，所以每次启动只派发一种标量类型；结果始终是
// INT8。
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace ops {

Tensor EqualImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpEqual, a, b);
}
Tensor NotEqualImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpNotEqual, a, b);
}
Tensor GreaterImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpGreater, a, b);
}
Tensor LessImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpLess, a, b);
}
Tensor GreaterEqualImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpGreaterEqual, a, b);
}
Tensor LessEqualImpl<Device::SYCL>::execute(const Tensor& a, const Tensor& b) {
    return sycl_ufunc::compare_impl(tvv::sycl::kCmpLessEqual, a, b);
}

Tensor EqualImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpEqual, a, val);
}
Tensor NotEqualImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpNotEqual, a, val);
}
Tensor GreaterImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpGreater, a, val);
}
Tensor LessImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpLess, a, val);
}
Tensor GreaterEqualImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpGreaterEqual, a, val);
}
Tensor LessEqualImpl<Device::SYCL>::execute(const Tensor& a, float val) {
    return sycl_ufunc::compare_scalar(tvv::sycl::kCmpLessEqual, a, val);
}

}  // namespace ops
