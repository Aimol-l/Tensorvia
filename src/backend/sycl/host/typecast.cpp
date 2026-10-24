// Element-wise dtype conversion for Device::SYCL.
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "sycl_island.h"

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/sycl/sycl_ops.hpp"
#include "interface/backend/sycl/sycl_ufunc.hpp"

namespace ops {

namespace {

Tensor cast_tensor(const Tensor& a, DataType dst_type) {
    // The kernel reads a flat buffer; materialize any strided view.
    const Tensor src = a.is_contiguous() ? a : a.clone();
    Tensor out(src.shape(), dst_type, Device::SYCL);

    auto ctx = sycl_ufunc::context_of(src, "Typecast");
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_cast(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype()), static_cast<int>(dst_type)),
                      "Typecast launch");
    ctx->wait();
    return out;
}

}  // namespace

void TypecastImpl<Device::SYCL>::execute(Tensor& a, DataType dst_type) {
    if (a.dtype() == dst_type) return;
    Tensor converted = cast_tensor(a, dst_type);
    a.set_impl(converted.get_impl(), dst_type);
}

Tensor TypecastImpl<Device::SYCL>::execute(const Tensor& a, DataType dst_type) {
    if (a.dtype() == dst_type) return a;
    return cast_tensor(a, dst_type);
}

}  // namespace ops
