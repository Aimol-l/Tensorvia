#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

std::shared_ptr<VKTensor> impl_of(const Tensor& t, const char* what) {
    auto impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    if (!impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
    return impl;
}

std::shared_ptr<VulkanContext> context_of(const Tensor& t, const char* what) {
    auto ctx = std::dynamic_pointer_cast<VulkanContext>(impl_of(t, what)->context());
    if (!ctx)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    return ctx;
}

int32_t to_dim(int64_t v, const char* what) {
    if (v < 0 || v > INT32_MAX)
        throw std::runtime_error(std::string("vulkan ") + what + ": dimension exceeds int32");
    return static_cast<int32_t>(v);
}

// LayerNorm 与 RmsNorm 只差均值项和一个仿射偏置；其余路径共用。
//
// 内核把输入按 [outer, norm_size] 的连续行主序解释，一个 workgroup 一行，
// 所以跨步视图先物化。只有 FLOAT32 注册了流水线——ops.cpp 只在 a.dtype() ==
// FLOAT32 时才派发到这里。
Tensor norm_impl(const Tensor& a, const Tensor* weight, const Tensor* bias, int axis,
                 float eps, bool rms, const char* what) {
    const int dim = static_cast<int>(a.shape().size());
    if (axis < 0) axis += dim;
    if (axis < 0 || axis >= dim)
        throw std::runtime_error(std::string("vulkan ") + what + ": axis out of range");
    if (a.dtype() != DataType::FLOAT32)
        throw std::runtime_error(std::string("vulkan ") + what +
                                 ": the fused kernel only covers FLOAT32");

    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= a.shape(i);
    int64_t n = 1;
    for (int i = axis; i < dim; ++i) n *= a.shape(i);
    if (n == 0 || outer == 0) {
        // 空输入由 ops.cpp 的分派层处理；这里只防御内核几何为 0 的派发。
        return Tensor(a.shape(), DataType::FLOAT32, Device::VULKAN);
    }

    const Tensor src = a.is_contiguous() ? a : a.clone();
    const Tensor w   = weight->is_contiguous() ? *weight : weight->clone();
    Tensor b_storage;
    const Tensor* b = bias;
    if (b != nullptr && !b->is_contiguous()) {
        b_storage = b->clone();
        b         = &b_storage;
    }

    const auto a_shape = a.shape();
    Tensor out(std::vector<int64_t>(a_shape.begin(), a_shape.end()), DataType::FLOAT32,
               Device::VULKAN);

    auto ctx = context_of(src, what);
    std::vector<vk::Buffer> buffers{impl_of(src, what)->buffer(), impl_of(w, what)->buffer()};
    if (b != nullptr) buffers.push_back(impl_of(*b, what)->buffer());
    buffers.push_back(impl_of(out, what)->buffer());

    NormParams p{.norm_size = to_dim(n, what), .outer = to_dim(outer, what), .eps = eps};
    ctx->submitCompute(rms ? OpType::RmsNorm : OpType::LayerNorm, DataType::FLOAT32, buffers,
                       static_cast<uint32_t>(outer), 1, 1, &p, sizeof(p));
    return out;
}

}  // namespace

Tensor LayerNormImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& weight,
                                              const Tensor& bias, int axis, float eps) {
    return norm_impl(a, &weight, &bias, axis, eps, false, "LayerNorm");
}

Tensor RmsNormImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& weight, int axis,
                                            float eps) {
    return norm_impl(a, &weight, nullptr, axis, eps, true, "RmsNorm");
}

}  // namespace ops
