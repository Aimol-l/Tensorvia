// Global and per-axis sum/min/max/mean, argmax/argmin, all/any and the NonZero
// count — the Device::SYCL host driver.
//
// The kernels index a flat row-major buffer, so a strided view is materialized
// once here. Tensor::data() (not the impl's base pointer) is passed on, so a
// contiguous view with a non-zero offset reads its own elements.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_island.h"

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/sycl/sycl_ops.hpp"
#include "interface/backend/sycl/sycl_context.hpp"
#include "interface/backend/sycl/sycl_tensor.hpp"
#include "interface/backend/sycl/sycl_ufunc.hpp"

namespace ops {

namespace {

std::shared_ptr<SYCLContext> context_of(const Tensor& t, const char* what) {
    const auto impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("sycl ") + what +
                                 ": operand does not live on the SYCL device");
    auto ctx = std::dynamic_pointer_cast<SYCLContext>(impl->context());
    if (!ctx) throw std::runtime_error(std::string("sycl ") + what + ": missing SYCLContext");
    return ctx;
}

Tensor materialize(const Tensor& a) {
    return a.is_contiguous() ? a : a.clone();
}

struct AxisLayout {
    tvv::sycl::ReduceAxis params{};
    std::vector<int64_t> shape;
};

AxisLayout axis_layout(const Tensor& a, int axis, const char* what) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error(std::string("sycl ") + what + ": axis out of range");

    AxisLayout layout;
    layout.params.axis  = a.shape(axis);
    layout.params.inner = 1;
    for (int i = axis + 1; i < dim; ++i) layout.params.inner *= a.shape(i);
    layout.params.outer = 1;
    for (int i = 0; i < axis; ++i) layout.params.outer *= a.shape(i);

    for (int i = 0; i < dim; ++i) {
        if (i != axis) layout.shape.push_back(a.shape(i));
    }
    // Dropping the only dimension of a 1D input leaves rank 0, which tensors do
    // not support; {1} matches the CPU/Vulkan fallback.
    if (layout.shape.empty()) layout.shape = {1};
    return layout;
}

// Global reductions fold into one FLOAT64 element on the device and are
// narrowed to float on the host, which is the API's return type.
double reduce_all_value(const Tensor& a, int op, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    Tensor out({1}, DataType::FLOAT64, Device::SYCL);

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_reduce_all(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype()),
                          static_cast<int>(DataType::FLOAT64), op),
                      what);
    ctx->wait();

    out.to_host();
    return static_cast<const double*>(out.data())[0];
}

Tensor axis_reduce(const Tensor& a, int axis, DataType out_type, int op, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    const AxisLayout layout = axis_layout(a, axis, what);

    Tensor out(layout.shape, out_type, Device::SYCL);
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_reduce_axis(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int>(src.dtype()), static_cast<int>(out_type),
                          op, &layout.params),
                      what);
    ctx->wait();
    return out;
}

Tensor arg_reduce(const Tensor& a, int axis, bool is_max, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    const AxisLayout layout = axis_layout(a, axis, what);

    Tensor out(layout.shape, DataType::INT32, Device::SYCL);
    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_arg_reduce(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int>(src.dtype()), is_max ? 1 : 0,
                          &layout.params),
                      what);
    ctx->wait();
    return out;
}

bool all_any_value(const Tensor& a, float value, bool is_all, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    // The kernel initializes this word itself (init launch, then atomics), so
    // it does not need a host round-trip before the reduction.
    Tensor out({1}, DataType::INT32, Device::SYCL);

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_all_any(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype()), is_all ? 1 : 0,
                          static_cast<double>(value)),
                      what);
    ctx->wait();

    out.to_host();
    return static_cast<const int32_t*>(out.data())[0] != 0;
}

}  // namespace

float SumImpl<Device::SYCL>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::sycl::kReduceSum, "Sum"));
}

Tensor SumImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    // Sum widens narrow integers and 16-bit floats to a real accumulator type;
    // the Vulkan backend uses the same rule (the CPU one keeps the input dtype).
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::sycl::kReduceSum, "Sum");
}

float MinImpl<Device::SYCL>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::sycl::kReduceMin, "Min"));
}

Tensor MinImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::sycl::kReduceMin, "Min");
}

float MaxImpl<Device::SYCL>::execute(const Tensor& a) {
    return static_cast<float>(reduce_all_value(a, tvv::sycl::kReduceMax, "Max"));
}

Tensor MaxImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::sycl::kReduceMax, "Max");
}

float MeanImpl<Device::SYCL>::execute(const Tensor& a) {
    // Divide in double: above 2^24 a float divisor is no longer exact.
    const double sum = reduce_all_value(a, tvv::sycl::kReduceSum, "Mean");
    return static_cast<float>(sum / static_cast<double>(a.numel()));
}

Tensor MeanImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, float_reduction_type(a.dtype()),
                       tvv::sycl::kReduceMean, "Mean");
}

Tensor ArgMaxImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return arg_reduce(a, axis, true, "Argmax");
}

Tensor ArgMinImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return arg_reduce(a, axis, false, "Argmin");
}

bool AllImpl<Device::SYCL>::execute(const Tensor& a, float value) {
    return all_any_value(a, value, true, "All");
}

bool AnyImpl<Device::SYCL>::execute(const Tensor& a, float value) {
    return all_any_value(a, value, false, "Any");
}

size_t NonZeroImpl<Device::SYCL>::execute(const Tensor& a) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, "NonZero");
    // The seam's atomic counter is a 32-bit word (OpenCL 1.2 has no 64-bit
    // atomics); the host widens the count to size_t.
    Tensor out({1}, DataType::INT32, Device::SYCL);

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_nonzero(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype())),
                      "NonZero launch");
    ctx->wait();

    out.to_host();
    return static_cast<size_t>(static_cast<const int32_t*>(out.data())[0]);
}

}  // namespace ops
