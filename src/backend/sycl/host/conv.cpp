// Device::SYCL 的 2-D 卷积。
//
// 几何形状与 CPU 实现逐项对齐：输入/输出都是连续 NCHW，设备内核按扁平输出下标
// 解码 (n, oc, oy, ox)。跨步视图在这里物化成连续布局。
#include <cstdint>
#include <memory>
#include <vector>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
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

Tensor Conv2dImpl<Device::SYCL>::execute(const Tensor& input, const Tensor& weight,
                                         const Tensor* bias, const Conv2dShape& shape) {
    const Tensor src = materialize(input);
    const Tensor w   = materialize(weight);

    Tensor result({shape.batch, shape.out_ch, shape.out_h, shape.out_w}, src.dtype(),
                  Device::SYCL);

    const auto ctx = context_of(src, "conv2d");

    const void* bias_ptr = nullptr;
    if (bias != nullptr) bias_ptr = base_of(*bias, "conv2d bias");

    const tvv::sycl::Conv2d p{.batch   = shape.batch,
                              .in_ch   = shape.in_ch,
                              .in_h    = shape.in_h,
                              .in_w    = shape.in_w,
                              .out_ch  = shape.out_ch,
                              .kh      = shape.kh,
                              .kw      = shape.kw,
                              .out_h   = shape.out_h,
                              .out_w   = shape.out_w,
                              .stride  = static_cast<int32_t>(shape.stride),
                              .padding = static_cast<int32_t>(shape.padding)};

    check(::sycl_c_launch_conv2d(handle_of(*ctx), base_of(src, "conv2d"),
                                 base_of(w, "conv2d weight"), bias_ptr,
                                 mutable_base_of(result, "conv2d"), static_cast<int>(src.dtype()),
                                 &p),
          "conv2d launch");
    return result;
}

}  // namespace ops
