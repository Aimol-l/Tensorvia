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
    // 输入的真实元素数：轴循环按它截断（见 reduce_axis_kernel）。正常布局下等于
    // outer*axis*inner，填真值只是让恒等式显式。
    layout.params.numel = static_cast<int64_t>(a.numel());

    for (int i = 0; i < dim; ++i) {
        if (i != axis) layout.shape.push_back(a.shape(i));
    }
    // 单轴内核早于 rank-0 张量出现，把空的归约形状写作 {1}；核心层通过
    // with_reduced_shape 把结果重塑为真正的 0-dim 张量，所以该回退永远不会到达
    // 公开 API。
    if (layout.shape.empty()) layout.shape = {1};
    return layout;
}

// 全归约的两趟派发。
//
// 轴内核的几何是 outer x axis x inner，全归约展平后是 outer=inner=1、
// axis=numel，于是 total=1：一个 work-group 里只有一个工作项在跑。改成先切成
// kFlatReduceBlocks 段并行算部分值，再对这 B 个部分值折叠。段长取 ceil，末段
// 短于段长，由内核按 numel 截断。
//
// B 取 1024 足以饱和常见设备；每段 N/1024 个元素（16M 时 64 个/工作项）仍是
// 访存受限而非 launch 受限。阈值以下不划算：单工作项串行只有数十 µs，多一次
// 提交加一次 scratch 分配回不来。
constexpr int64_t kFlatReduceBlocks   = 1024;
constexpr int64_t kFlatReduceMinNumel = 1 << 16;

bool flat_reduce_applies(const tvv::sycl::ReduceAxis& p) {
    return p.outer * p.inner == 1 && p.axis >= kFlatReduceMinNumel;
}

Tensor axis_reduce(const Tensor& a, int axis, DataType out_type, int op, const char* what) {
    const Tensor src = materialize(a);
    auto ctx = context_of(src, what);
    const AxisLayout layout = axis_layout(a, axis, what);

    Tensor out(layout.shape, out_type, Device::SYCL);
    ctx->activate();
    const auto handle = sycl_ufunc::handle_of(*ctx);
    const tvv::sycl::ReduceAxis& p = layout.params;

    if (flat_reduce_applies(p)) {
        const int64_t chunk = (p.axis + kFlatReduceBlocks - 1) / kFlatReduceBlocks;
        Tensor partial({kFlatReduceBlocks}, out_type, Device::SYCL);

        // 趟 1：Mean 先算纯部分和——末段短于 chunk，先除会少算，分母留到趟 2。
        const int pass1_op = (op == tvv::sycl::kReduceMean) ? tvv::sycl::kReduceSum : op;
        const tvv::sycl::ReduceAxis first{.outer = kFlatReduceBlocks,
                                          .axis = chunk,
                                          .inner = 1,
                                          .numel = p.axis,
                                          .mean_n = 0};
        sycl_ufunc::check(::sycl_c_launch_reduce_axis(handle, src.data(), partial.data(),
                                                     static_cast<int>(src.dtype()),
                                                     static_cast<int>(out_type), pass1_op,
                                                     &first),
                          what);

        // 趟 2：折叠 B 个部分值。输入就是 partial，元素数恰好 B，截断恒等。
        const tvv::sycl::ReduceAxis second{.outer = 1,
                                           .axis = kFlatReduceBlocks,
                                           .inner = 1,
                                           .numel = kFlatReduceBlocks,
                                           .mean_n = (op == tvv::sycl::kReduceMean) ? p.axis : 0};
        sycl_ufunc::check(::sycl_c_launch_reduce_axis(handle, partial.data(), out.data(),
                                                      static_cast<int>(out_type),
                                                      static_cast<int>(out_type), op, &second),
                          what);
        return out;
    }

    sycl_ufunc::check(::sycl_c_launch_reduce_axis(handle, src.data(), out.data(),
                                                  static_cast<int>(src.dtype()),
                                                  static_cast<int>(out_type), op, &p),
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
