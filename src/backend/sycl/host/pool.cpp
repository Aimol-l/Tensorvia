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

Tensor pool2d_execute(const Tensor& input, const Pool2dShape& shape, bool average) {
    const Tensor src = materialize(input);

    Tensor result({shape.batch, shape.channels, shape.out_h, shape.out_w}, src.dtype(),
                  Device::SYCL);

    const auto ctx = context_of(src, average ? "avg_pool2d" : "max_pool2d");

    const tvv::sycl::Pool2d p{.batch    = shape.batch,
                              .channels = shape.channels,
                              .in_h     = shape.in_h,
                              .in_w     = shape.in_w,
                              .kh       = shape.kh,
                              .kw       = shape.kw,
                              .stride_h = shape.stride_h,
                              .stride_w = shape.stride_w,
                              .padding  = shape.padding,
                              .out_h    = shape.out_h,
                              .out_w    = shape.out_w};

    check(::sycl_c_launch_pool2d(handle_of(*ctx), base_of(src, "pool2d"),
                                 mutable_base_of(result, "pool2d"),
                                 static_cast<int>(src.dtype()), &p, average ? 1 : 0),
          "pool2d launch");
    return result;
}

}  // namespace

Tensor MaxPool2dImpl<Device::SYCL>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, false);
}

Tensor AvgPool2dImpl<Device::SYCL>::execute(const Tensor& input, const Pool2dShape& shape) {
    return pool2d_execute(input, shape, true);
}

}  // namespace ops
