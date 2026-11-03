#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "sycl_island.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/tensor_iterator.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ufunc.hpp"

namespace {

using ops::sycl_ufunc::check;
using ops::sycl_ufunc::context_of;
using ops::sycl_ufunc::handle_of;
using ops::sycl_ufunc::require_dims;

// 内核线性遍历连续布局，所以跨步视图先一次物化。
Tensor materialize(const Tensor& t) { return t.is_contiguous() ? t : t.clone(); }

// 索引越界的契约来自 CPU：越界即刻抛 "index out of range"。设备端没有异常，所以
// 用一个标志 + 4 字节回读代替。这一次同步是这类算子唯一需要宿主参与的地方
//（NonZero 读计数同理），换来的是三个后端同一份错误行为。
void require_indices_in_range(Tensor flag, const char* op) {
    flag.to_host();
    if (static_cast<const int32_t*>(flag.data())[0] != 0)
        throw std::runtime_error(std::string(op) + ": index out of range");
}

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

// 索引越界标志：与其它设备后端一样，由设备内核置位、宿主回读。
Tensor make_flag() {
    return ops::Zeros(std::vector<int64_t>{1}, DataType::INT32, Device::SYCL);
}

}  // namespace

namespace ops {

Tensor WhereImpl<Device::SYCL>::execute(const Tensor& cond, const Tensor& a,
                                        const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor pa, pb;
    const Tensor* A = &a;
    const Tensor* B = &b;
    if (a.dtype() != res_type) { pa = ops::Typecast(a, res_type); A = &pa; }
    if (b.dtype() != res_type) { pb = ops::Typecast(b, res_type); B = &pb; }

    const Tensor src = materialize(cond);
    const Tensor Ac  = materialize(*A);
    const Tensor Bc  = materialize(*B);

    const iter::PreparedInputs in({&src, &Ac, &Bc}, "where");
    const iter::Layout& lay = in.layout();
    Tensor result(std::span<const int64_t>(lay.shape), res_type, Device::SYCL);

    require_dims(static_cast<size_t>(lay.ndim), "where");
    ::tvv::sycl::Select p{};
    p.ndim       = lay.ndim;
    p.numel      = lay.numel;
    p.cond_dtype = static_cast<int>(in[0].dtype());
    for (int32_t d = 0; d < lay.ndim; ++d) {
        p.shape[d] = lay.shape[d];
        p.sc[d]    = lay.operands[0].strides[d];
        p.sa[d]    = lay.operands[1].strides[d];
        p.sb[d]    = lay.operands[2].strides[d];
    }
    p.off_c = lay.operands[0].offset;
    p.off_a = lay.operands[1].offset;
    p.off_b = lay.operands[2].offset;
    p.off_o = result.offset();

    auto ctx = context_of(Ac, "where");
    ctx->activate();
    check(::sycl_c_launch_select(handle_of(*ctx), base_of(src, "where"),
                                 base_of(Ac, "where"), base_of(Bc, "where"),
                                 mutable_base_of(result, "where"),
                                 static_cast<int>(res_type), &p),
          "where launch");
    return result;
}

// 标量版本：把标量铺成 0 维张量，与张量-张量路径共用同一个内核。用 0 维而不是
// {1}：全 0 维输入的广播结果必须仍是 0 维（CPU 如此）。
Tensor WhereImpl<Device::SYCL>::execute(const Tensor& cond, const Tensor& a, float b) {
    const DataType res_type = scalar_promote_type(a.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, b, Device::SYCL);
    return WhereImpl<Device::SYCL>::execute(cond, a, scalar);
}

Tensor WhereImpl<Device::SYCL>::execute(const Tensor& cond, float a, const Tensor& b) {
    const DataType res_type = scalar_promote_type(b.dtype());
    const Tensor scalar = ops::Fill(std::vector<int64_t>{}, res_type, a, Device::SYCL);
    return WhereImpl<Device::SYCL>::execute(cond, scalar, b);
}

Tensor IndexSelectImpl<Device::SYCL>::execute(const Tensor& input, int dim,
                                              const Tensor& index) {
    const Tensor src = materialize(input);
    const Tensor idx = materialize(index);

    const auto in_shape = src.shape();
    const int64_t idx_len = static_cast<int64_t>(idx.numel());

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end());
    out_shape[static_cast<size_t>(dim)] = idx_len;
    Tensor result(out_shape, input.dtype(), Device::SYCL);

    ::tvv::sycl::IndexAxis p{};
    p.axis_len = in_shape[static_cast<size_t>(dim)];
    p.idx_len  = idx_len;
    p.inner    = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < in_shape.size(); ++i)
        p.inner *= in_shape[i];
    p.outer = 1;
    for (int i = 0; i < dim; ++i) p.outer *= in_shape[static_cast<size_t>(i)];

    Tensor flag = make_flag();
    auto ctx = context_of(src, "index_select");
    ctx->activate();
    check(::sycl_c_launch_index_select(
              handle_of(*ctx), base_of(src, "index_select"), base_of(idx, "index_select"),
              mutable_base_of(result, "index_select"), static_cast<int>(src.dtype()),
              static_cast<int>(idx.dtype()), &p,
              static_cast<int32_t*>(mutable_base_of(flag, "index_select"))),
          "index_select launch");
    require_indices_in_range(flag, "index_select");
    return result;
}

Tensor GatherImpl<Device::SYCL>::execute(const Tensor& input, int dim,
                                         const Tensor& index) {
    const Tensor src = materialize(input);
    const Tensor idx = materialize(index);

    const auto in_shape = src.shape();
    const auto idx_shape = idx.shape();
    const size_t ndim = in_shape.size();

    Tensor result(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()), input.dtype(),
                  Device::SYCL);

    require_dims(ndim, "gather");
    ::tvv::sycl::Gather p{};
    p.ndim  = static_cast<int32_t>(ndim);
    p.numel = static_cast<int64_t>(result.numel());
    p.dim   = dim;
    p.axis_dim = in_shape[static_cast<size_t>(dim)];
    int64_t acc = 1;
    for (size_t d = ndim; d-- > 0;) {
        p.in_strides[d] = acc;
        acc *= in_shape[d];
    }
    p.axis_stride = p.in_strides[static_cast<size_t>(dim)];
    for (size_t d = 0; d < ndim; ++d) p.out_shape[d] = idx_shape[d];

    Tensor flag = make_flag();
    auto ctx = context_of(src, "gather");
    ctx->activate();
    check(::sycl_c_launch_gather(
              handle_of(*ctx), base_of(src, "gather"), base_of(idx, "gather"),
              mutable_base_of(result, "gather"), static_cast<int>(src.dtype()),
              static_cast<int>(idx.dtype()), &p,
              static_cast<int32_t*>(mutable_base_of(flag, "gather"))),
          "gather launch");
    require_indices_in_range(flag, "gather");
    return result;
}

Tensor CumsumImpl<Device::SYCL>::execute(const Tensor& a, int axis) {
    const Tensor src = materialize(a);
    Tensor result(a.shape(), a.dtype(), Device::SYCL);

    ::tvv::sycl::IndexAxis p{};
    const auto shape = src.shape();
    p.axis_len = shape[static_cast<size_t>(axis)];
    p.inner    = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i)
        p.inner *= shape[i];
    p.outer = 1;
    for (int i = 0; i < axis; ++i) p.outer *= shape[static_cast<size_t>(i)];

    auto ctx = context_of(src, "cumsum");
    ctx->activate();
    check(::sycl_c_launch_cumsum(handle_of(*ctx), base_of(src, "cumsum"),
                                 mutable_base_of(result, "cumsum"),
                                 static_cast<int>(src.dtype()), &p),
          "cumsum launch");
    return result;
}

}  // namespace ops
