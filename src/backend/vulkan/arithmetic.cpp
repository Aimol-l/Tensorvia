#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"
#include "tensorvia/backend/vulkan/vulkan_ufunc.hpp"

namespace ops {

// ---------------------------------------------------------------------------
// 标量，原地。
// ---------------------------------------------------------------------------
void AddImpl<Device::VULKAN>::execute(Tensor& a, float b) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Add: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Add: missing VulkanContext");
    ValueParams<float> params{.value = b, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Add, a.dtype(), {src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

void SubImpl<Device::VULKAN>::execute(Tensor& a, float b) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Sub: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Sub: missing VulkanContext");
    ValueParams<float> params{.value = b, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Sub, a.dtype(), {src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

void MulImpl<Device::VULKAN>::execute(Tensor& a, float b) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Mul: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Mul: missing VulkanContext");
    ValueParams<float> params{.value = b, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Mul, a.dtype(), {src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

void DivImpl<Device::VULKAN>::execute(Tensor& a, float b) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Div: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Div: missing VulkanContext");
    ValueParams<float> params{.value = b, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Div, a.dtype(), {src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

// ---------------------------------------------------------------------------
// 标量，非原地。
// ---------------------------------------------------------------------------
Tensor AddImpl<Device::VULKAN>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    AddImpl<Device::VULKAN>::execute(t, b);
    return t;
}

Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, float b) {
    Tensor t = a.clone();
    SubImpl<Device::VULKAN>::execute(t, b);
    return t;
}

Tensor MulImpl<Device::VULKAN>::execute(const Tensor& a, float b) {
    Tensor t = ops::Fill(std::vector<int64_t>(a.shape().begin(), a.shape().end()), a.dtype(), b,
                         Device::VULKAN);
    return MulImpl<Device::VULKAN>::execute(a, t);
}

Tensor DivImpl<Device::VULKAN>::execute(const Tensor& a, float b) {
    Tensor t = ops::Fill(std::vector<int64_t>(a.shape().begin(), a.shape().end()), a.dtype(), b,
                         Device::VULKAN);
    return DivImpl<Device::VULKAN>::execute(a, t);
}

// ---------------------------------------------------------------------------
// 张量-张量：ufunc 广播驱动（dtype 提升 + NumPy 广播）。
// ---------------------------------------------------------------------------
Tensor AddImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::binary_arith(OpType::AddVec, a, b, nullptr);
}
Tensor SubImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::binary_arith(OpType::SubVec, a, b, nullptr);
}
Tensor MulImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::binary_arith(OpType::MulVec, a, b, nullptr);
}
Tensor DivImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    return vk_ufunc::binary_arith(OpType::DivVec, a, b, nullptr);
}

void AddImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    vk_ufunc::binary_arith(OpType::AddVec, a, b, &dst);
}
void SubImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    vk_ufunc::binary_arith(OpType::SubVec, a, b, &dst);
}
void MulImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    vk_ufunc::binary_arith(OpType::MulVec, a, b, &dst);
}
void DivImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b, Tensor& dst) {
    vk_ufunc::binary_arith(OpType::DivVec, a, b, &dst);
}

// ---------------------------------------------------------------------------
// 一元逐元素运算。
// ---------------------------------------------------------------------------
void AbsImpl<Device::VULKAN>::execute(Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Abs: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Abs: missing VulkanContext");
    int64_t numel = a.numel();
    // abs 有两个绑定（读/写）；原地意味着二者是同一个 buffer。
    ctx_impl->submitCompute(OpType::Abs, a.dtype(),
                            {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

namespace {
template <class Impl>
void unary_inplace(Tensor& a, OpType op, const char* what) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    int64_t numel = a.numel();
    ctx_impl->submitCompute(op, a.dtype(), {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
}

// 非原地一元：直接写一块新分配的 dst，而不是 clone() 后跑原地内核。clone 那条路径
// 要把整个张量多读多写一遍（16M f32 是 128 MB 的额外流量），而着色器本来就支持
// 读 0 写 1 两个绑定。
template <class Impl>
Tensor unary_out_of_place(const Tensor& a, OpType op, const char* what) {
    // int8/16/32 -> float32；int64 -> float64；浮点保持自身 dtype。
    DataType rt = a.dtype();
    if (is_integral_dtype(rt)) rt = DataType::FLOAT32;
    else if (rt == DataType::INT64) rt = DataType::FLOAT64;

    Tensor storage;
    const Tensor* src = vulkan_detail::unary_source(a, rt, storage);
    Tensor result(a.shape(), rt, Device::VULKAN);

    auto src_impl = std::dynamic_pointer_cast<VKTensor>(src->get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    int64_t numel = result.numel();
    ctx_impl->submitCompute(op, rt, {src_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

template <class Impl>
void value_inplace(Tensor& a, OpType op, float val, const char* what) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    ValueParams<float> params{.value = val, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(op, a.dtype(), {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

template <class Impl>
Tensor value_out_of_place(const Tensor& a, OpType op, float val, const char* what) {
    DataType rt = a.dtype();
    if (is_integral_dtype(rt)) rt = DataType::FLOAT32;
    else if (rt == DataType::INT64) rt = DataType::FLOAT64;

    Tensor storage;
    const Tensor* src = vulkan_detail::unary_source(a, rt, storage);
    Tensor result(a.shape(), rt, Device::VULKAN);

    auto src_impl = std::dynamic_pointer_cast<VKTensor>(src->get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    ValueParams<float> params{.value = val, .numel = static_cast<int64_t>(result.numel())};
    ctx_impl->submitCompute(op, rt, {src_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &params, sizeof(params));
    return result;
}
}  // namespace

void SinImpl<Device::VULKAN>::execute(Tensor& a) {
    unary_inplace<SinImpl<Device::VULKAN>>(a, OpType::Sin, "Sin");
}
void CosImpl<Device::VULKAN>::execute(Tensor& a) {
    unary_inplace<CosImpl<Device::VULKAN>>(a, OpType::Cos, "Cos");
}
void TanImpl<Device::VULKAN>::execute(Tensor& a) {
    unary_inplace<TanImpl<Device::VULKAN>>(a, OpType::Tan, "Tan");
}
void ExpImpl<Device::VULKAN>::execute(Tensor& a) {
    unary_inplace<ExpImpl<Device::VULKAN>>(a, OpType::Exp, "Exp");
}
void SqrtImpl<Device::VULKAN>::execute(Tensor& a) {
    unary_inplace<SqrtImpl<Device::VULKAN>>(a, OpType::Sqrt, "Sqrt");
}

void PowImpl<Device::VULKAN>::execute(Tensor& a, float val) {
    value_inplace<PowImpl<Device::VULKAN>>(a, OpType::Pow, val, "Pow");
}
void LogImpl<Device::VULKAN>::execute(Tensor& a, float val) {
    value_inplace<LogImpl<Device::VULKAN>>(a, OpType::Log, val, "Log");
}

void ClampImpl<Device::VULKAN>::execute(Tensor& a, float min, float max) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    if (!src_impl) throw std::runtime_error("vulkan Clamp: expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Clamp: missing VulkanContext");
    ClampParams params{.min = min, .max = max, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Clamp, a.dtype(),
                            {src_impl->buffer(), src_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
}

Tensor AbsImpl<Device::VULKAN>::execute(const Tensor& a) {
    // abs 收 AllTypes 且不改 dtype（不像 sin 那样把整型提到浮点），所以没有提升
    // 这一步：源直接取原张量（视图先物化），一次 src -> dst。
    Tensor storage;
    const Tensor& src = vulkan_detail::linear_source(a, storage);
    Tensor result(a.shape(), a.dtype(), Device::VULKAN);
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(src.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error("vulkan Abs: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Abs: missing VulkanContext");
    int64_t numel = result.numel();
    ctx_impl->submitCompute(OpType::Abs, result.dtype(),
                            {src_impl->buffer(), dst_impl->buffer()},
                            (result.numel() + 255) / 256, 1, 1, &numel, sizeof(int64_t));
    return result;
}

Tensor SinImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<SinImpl<Device::VULKAN>>(a, OpType::Sin, "Sin");
}
Tensor CosImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<CosImpl<Device::VULKAN>>(a, OpType::Cos, "Cos");
}
Tensor TanImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<TanImpl<Device::VULKAN>>(a, OpType::Tan, "Tan");
}
Tensor ExpImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<ExpImpl<Device::VULKAN>>(a, OpType::Exp, "Exp");
}

Tensor ErfImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<ErfImpl<Device::VULKAN>>(a, OpType::Erf, "Erf");
}
Tensor SqrtImpl<Device::VULKAN>::execute(const Tensor& a) {
    return unary_out_of_place<SqrtImpl<Device::VULKAN>>(a, OpType::Sqrt, "Sqrt");
}
Tensor PowImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return value_out_of_place<PowImpl<Device::VULKAN>>(a, OpType::Pow, val, "Pow");
}
Tensor LogImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    return value_out_of_place<LogImpl<Device::VULKAN>>(a, OpType::Log, val, "Log");
}

Tensor ClampImpl<Device::VULKAN>::execute(const Tensor& a, float min, float max) {
    Tensor result(a.shape(), a.dtype(), a.device());
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error("vulkan Clamp: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Clamp: missing VulkanContext");
    ClampParams params{.min = min, .max = max, .numel = static_cast<int64_t>(a.numel())};
    ctx_impl->submitCompute(OpType::Clamp, a.dtype(),
                            {src_impl->buffer(), dst_impl->buffer()},
                            (a.numel() + 255) / 256, 1, 1, &params, sizeof(params));
    return result;
}

}  // namespace ops
