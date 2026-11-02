// 单轴的 sum/min/max/mean、argmax/argmin、all/any 以及 NonZero 计数
// ——Device::SYCL 宿主驱动。
//
// 内核索引扁平行主序 buffer，所以视图在这里一次性物化。传下去的是
// Tensor::data()（而不是 impl 的基址指针），因此带非零 offset 的连续视图读取的是
// 自己的元素。
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
    // 单轴内核早于 rank-0 张量出现，把空的归约形状写作 {1}；核心层通过
    // with_reduced_shape 把结果重塑为真正的 0-dim 张量，所以该回退永远不会到达
    // 公开 API。
    if (layout.shape.empty()) layout.shape = {1};
    return layout;
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
    return out;
}

bool all_any_value(const Tensor& a, float value, bool is_all, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    // 内核自己初始化这个字（先 init 启动，再原子操作），所以归约前无需宿主往返。
    Tensor out({1}, DataType::INT32, Device::SYCL);

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_all_any(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype()), is_all ? 1 : 0,
                          static_cast<double>(value)),
                      what);

    out.to_host();
    return static_cast<const int32_t*>(out.data())[0] != 0;
}

}  // namespace

Tensor SumImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    // Sum 把窄整型与 16 位浮点拓宽到真正的累加类型；所有后端遵循同一条
    // reduction_type 规则。
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::sycl::kReduceSum, "Sum");
}

Tensor MinImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::sycl::kReduceMin, "Min");
}

Tensor MaxImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    return axis_reduce(a, axis, a.dtype(), tvv::sycl::kReduceMax, "Max");
}

Tensor ProdImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    // Prod 与 Sum 同族：窄整型与 16 位浮点拓宽到 reduction_type 再累加。
    return axis_reduce(a, axis, reduction_type(a.dtype()), tvv::sycl::kReduceProd, "Prod");
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
    // 接缝的原子计数器是 32 位字（OpenCL 1.2 没有 64 位原子操作）；宿主再把计数
    // 拓宽到 size_t。
    Tensor out({1}, DataType::INT32, Device::SYCL);

    ctx->activate();
    sycl_ufunc::check(::sycl_c_launch_nonzero(
                          sycl_ufunc::handle_of(*ctx), src.data(), out.data(),
                          static_cast<int64_t>(src.numel()),
                          static_cast<int>(src.dtype())),
                      "NonZero launch");

    out.to_host();
    return static_cast<size_t>(static_cast<const int32_t*>(out.data())[0]);
}

}  // namespace ops
