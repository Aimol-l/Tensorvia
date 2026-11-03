#include <memory>
#include <stdexcept>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
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

    // Softmax 每次只读写一条归约轴；内核处理扁平行主序 buffer，所以跨步操作数要先
    // 物化。
    const Tensor src = a.is_contiguous() ? a : a.clone();
    tvv::sycl::ReduceAxis p{};
    p.axis  = a.shape(axis);
    p.inner = 1;
    for (int i = axis + 1; i < dim; ++i) p.inner *= a.shape(i);
    p.outer = 1;
    for (int i = 0; i < axis; ++i) p.outer *= a.shape(i);

    // 整数没有有意义的概率输出；其它后端将其提升到 FLOAT32，其余类型保持其存储
    // 类型。
    const DataType out_type =
        is_integral_dtype(src.dtype()) ? DataType::FLOAT32 : src.dtype();
    Tensor out(src.shape(), out_type, Device::SYCL);

    auto ctx = sycl_ufunc::context_of(src, "Softmax");
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_softmax(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int>(src.dtype()), static_cast<int>(out_type), &p),
                      "Softmax launch");
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
