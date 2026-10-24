// Softmax (the only activation that reduces along an axis) and the host driver
// for Relu/Silu/Tanh/Sigmoid/Softmax on Device::SYCL.
//
// Relu/Silu/Tanh/Sigmoid reuse the element-wise unary path in :sycl_ufunc;
// Softmax needs outer x axis x inner geometry, so it fills a ReduceAxis.
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
#include "tensorvia/backend/sycl/sycl_context.hpp"
#include "tensorvia/backend/sycl/sycl_tensor.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace ops {

namespace {

Tensor softmax_impl(const Tensor& a, int axis) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error("sycl Softmax: axis out of range");

    // Softmax reads and writes one reduction axis at a time; the kernel works
    // on a flat row-major buffer, so a strided operand is materialized first.
    const Tensor src = a.is_contiguous() ? a : a.clone();
    tvv::sycl::ReduceAxis p{};
    p.axis  = a.shape(axis);
    p.inner = 1;
    for (int i = axis + 1; i < dim; ++i) p.inner *= a.shape(i);
    p.outer = 1;
    for (int i = 0; i < axis; ++i) p.outer *= a.shape(i);

    // Integers have no meaningful probability output; the other backends
    // promote them to FLOAT32, everything else keeps its storage type.
    const DataType out_type =
        src.dtype() <= DataType::INT64 ? DataType::FLOAT32 : src.dtype();
    Tensor out(src.shape(), out_type, Device::SYCL);

    auto ctx = sycl_ufunc::context_of(src, "Softmax");
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_softmax(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int>(src.dtype()), static_cast<int>(out_type), &p),
                      "Softmax launch");
    ctx->wait();
    return out;
}

}  // namespace

void ReluImpl<Device::SYCL>::execute(Tensor& a) {
    sycl_ufunc::unary_inplace(a, tvv::sycl::kUnaryRelu, 0.0f, 0.0f, "Relu");
}
Tensor ReluImpl<Device::SYCL>::execute(const Tensor& a) {
    return sycl_ufunc::unary_out(a, a.dtype(), tvv::sycl::kUnaryRelu, 0.0f, 0.0f, "Relu");
}

void SiluImpl<Device::SYCL>::execute(Tensor& a) {
    sycl_ufunc::unary_inplace(a, tvv::sycl::kUnarySilu, 0.0f, 0.0f, "Silu");
}
Tensor SiluImpl<Device::SYCL>::execute(const Tensor& a) {
    return sycl_ufunc::unary_out(a, sycl_ufunc::float_result(a.dtype()),
                                 tvv::sycl::kUnarySilu, 0.0f, 0.0f, "Silu");
}

void TanhImpl<Device::SYCL>::execute(Tensor& a) {
    sycl_ufunc::unary_inplace(a, tvv::sycl::kUnaryTanh, 0.0f, 0.0f, "Tanh");
}
Tensor TanhImpl<Device::SYCL>::execute(const Tensor& a) {
    return sycl_ufunc::unary_out(a, sycl_ufunc::float_result(a.dtype()),
                                 tvv::sycl::kUnaryTanh, 0.0f, 0.0f, "Tanh");
}

void SigmoidImpl<Device::SYCL>::execute(Tensor& a) {
    sycl_ufunc::unary_inplace(a, tvv::sycl::kUnarySigmoid, 0.0f, 0.0f, "Sigmoid");
}
Tensor SigmoidImpl<Device::SYCL>::execute(const Tensor& a) {
    return sycl_ufunc::unary_out(a, sycl_ufunc::float_result(a.dtype()),
                                 tvv::sycl::kUnarySigmoid, 0.0f, 0.0f, "Sigmoid");
}

Tensor SoftmaxImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return softmax_impl(a, axis);
}

}  // namespace ops
