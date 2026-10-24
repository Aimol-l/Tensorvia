#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

// Host-side ufunc driver for Vulkan: dtype promotion + NumPy broadcasting.
// The broadcast layout is derived by the device-independent helpers in
// :cpu_ops; strides and view offsets travel to the shader through
// BcastVecParams, so non-contiguous views do not need to be materialized.
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops::vk_ufunc {

using ops::ufunc::broadcast_shapes;
using ops::ufunc::broadcast_strides;
using ops::ufunc::same_shape;

// Fills BcastVecParams from the two operands (out_shape is the broadcast
// result shape). Strides and offsets are int32 element counts; out-of-range
// values are rejected, matching the repack kernel's 32-bit convention.
inline BcastVecParams make_bcast_params(const Tensor& A, const Tensor& B,
                                        std::span<const int64_t> out_shape) {
    BcastVecParams p{};
    p.ndim = int32_t(out_shape.size());
    int64_t total = 1;
    for (int64_t d : out_shape) total *= d;
    if (total > INT32_MAX)
        throw std::runtime_error("vulkan binary op: result exceeds 2^31 elements");
    p.numel = total;
    const auto sa = broadcast_strides(A.shape(), A.strides(), out_shape.size());
    const auto sb = broadcast_strides(B.shape(), B.strides(), out_shape.size());
    for (size_t d = 0; d < out_shape.size(); ++d) {
        if (out_shape[d] > INT32_MAX || sa[d] > INT32_MAX || sb[d] > INT32_MAX)
            throw std::runtime_error("vulkan binary op: shape/stride exceeds int32");
        p.shape[d] = int32_t(out_shape[d]);
        p.sa[d]    = int32_t(sa[d]);
        p.sb[d]    = int32_t(sb[d]);
    }
    if (A.offset() > INT32_MAX || B.offset() > INT32_MAX)
        throw std::runtime_error("vulkan binary op: view offset exceeds int32");
    p.off_a = int32_t(A.offset());
    p.off_b = int32_t(B.offset());
    return p;
}

// Zero-copy when no conversion is needed, otherwise a materialized Typecast
// temporary. submitCompute waits on its fence, so the temporary is fully
// consumed before it goes out of scope.
inline const Tensor& promote(const Tensor& t, DataType res, Tensor& storage) {
    if (t.dtype() == res) return t;
    storage = ops::Typecast(t, res);
    return storage;
}

// Generic broadcast binary submission. A/B already share a dtype; out_dtype
// is the output dtype (arithmetic = promoted dtype, logical = INT8).
// dst == nullptr allocates a new output; otherwise it validates and writes dst.
inline Tensor bcast_binary(OpType op, const Tensor& A, const Tensor& B,
                           DataType out_dtype, Tensor* dst) {
    DimArray out_shape;
    if (!broadcast_shapes(A.shape(), B.shape(), out_shape))
        throw std::runtime_error("binary op: shapes are not broadcastable");
    const std::span<const int64_t> want(out_shape);

    if (dst != nullptr) {
        if (!dst->is_contiguous())
            throw std::runtime_error(
                "dst: non-contiguous view is not supported here; call to_contiguous() first");
        if (dst->dtype() != out_dtype)
            throw std::runtime_error("dst dtype must equal the promoted result dtype");
        if (!same_shape(dst->shape(), want))
            throw std::runtime_error("dst shape must equal the broadcast result shape");
    }
    Tensor res;
    if (dst == nullptr) res = Tensor(want, out_dtype, Device::VULKAN);
    Tensor& out = dst != nullptr ? *dst : res;

    const auto A_impl   = std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    const auto B_impl   = std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    const auto out_impl = std::dynamic_pointer_cast<VKTensor>(out.get_impl());
    if (!A_impl || !B_impl || !out_impl)
        throw std::runtime_error("vulkan binary op: operands must live on the VULKAN device");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    if (!ctx_impl)
        throw std::runtime_error("vulkan binary op: missing VulkanContext");

    const BcastVecParams p = [&] {
        BcastVecParams q = make_bcast_params(A, B, want);
        if (out.offset() > INT32_MAX)
            throw std::runtime_error("vulkan binary op: view offset exceeds int32");
        q.off_o = int32_t(out.offset());
        return q;
    }();
    ctx_impl->submitCompute(
        op, A.dtype(),
        {A_impl->buffer(), B_impl->buffer(), out_impl->buffer()},
        static_cast<uint32_t>((p.numel + 255) / 256), 1, 1,
        &p, sizeof(p));
    return res;
}

// Binary arithmetic: dtype promotion + NumPy broadcasting.
inline Tensor binary_arith(OpType op, const Tensor& a, const Tensor& b, Tensor* dst) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, res_type, dst);
}

// Logical comparison: promote to a common dtype, compare, write INT8.
inline Tensor logical_impl(OpType op, const Tensor& a, const Tensor& b) {
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    Tensor storage_a, storage_b;
    const Tensor& A = promote(a, res_type, storage_a);
    const Tensor& B = promote(b, res_type, storage_b);
    return bcast_binary(op, A, B, DataType::INT8, nullptr);
}

// Scalar comparison: the right operand is a one-element FLOAT32 tensor.
inline Tensor logical_scalar(OpType op, const Tensor& a, float val) {
    const DataType res_type = compute_type(a.dtype(), DataType::FLOAT32);
    Tensor storage_a;
    const Tensor& A = promote(a, res_type, storage_a);
    Tensor scalar = ops::Fill(std::vector<int64_t>{1}, DataType::FLOAT32, val, Device::VULKAN);
    Tensor storage_b;
    const Tensor& B = promote(scalar, res_type, storage_b);
    return bcast_binary(op, A, B, DataType::INT8, nullptr);
}

}  // namespace ops::vk_ufunc
