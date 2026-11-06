#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace {

using ops::sycl_ufunc::check;
using ops::sycl_ufunc::context_of;
using ops::sycl_ufunc::handle_of;

Tensor materialize(const Tensor& t) { return t.is_contiguous() ? t : t.clone(); }

const void* base_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    return impl->data();
}

void* mutable_base_of(Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    return impl->data();
}

}  // namespace

namespace ops {

namespace {

Tensor conv_pool_execute(const Tensor& input, const Tensor& weight, const Tensor* bias,
                         const ConvPoolShape& shape) {
    const Tensor src = materialize(input);
    const Tensor w   = materialize(weight);

    Tensor result({shape.conv.batch, shape.conv.out_ch, shape.out_h, shape.out_w},
                  src.dtype(), Device::SYCL);

    const auto ctx = context_of(src, "conv_pool");

    const void* bias_ptr = nullptr;
    if (bias != nullptr) bias_ptr = base_of(*bias, "conv_pool bias");

    const tvv::sycl::ConvPool p{.batch   = shape.conv.batch,
                                .in_ch   = shape.conv.in_ch,
                                .in_h    = shape.conv.in_h,
                                .in_w    = shape.conv.in_w,
                                .out_ch  = shape.conv.out_ch,
                                .kh      = shape.conv.kh,
                                .kw      = shape.conv.kw,
                                .stride  = shape.conv.stride,
                                .padding = shape.conv.padding,
                                .conv_h  = shape.conv.out_h,
                                .conv_w  = shape.conv.out_w,
                                .out_h   = shape.out_h,
                                .out_w   = shape.out_w};

    check(::sycl_c_launch_conv_pool(handle_of(*ctx), base_of(src, "conv_pool"),
                                    base_of(w, "conv_pool weight"), bias_ptr,
                                    mutable_base_of(result, "conv_pool"),
                                    static_cast<int>(src.dtype()), &p),
          "conv_pool launch");
    return result;
}

}  // namespace

Tensor Conv2dReluMaxPool2dImpl<Device::SYCL>::execute(const Tensor& input,
                                                      const Tensor& weight,
                                                      const Tensor* bias,
                                                      const ConvPoolShape& shape) {
    return conv_pool_execute(input, weight, bias, shape);
}

}  // namespace ops
