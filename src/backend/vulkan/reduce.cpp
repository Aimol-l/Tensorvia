#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

inline void check_vk(const std::shared_ptr<TensorImpl>& impl, const char* what) {
    if (!impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
}

}  // namespace

Tensor SumImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Sum");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Sum: missing VulkanContext");

    const int dims = static_cast<int>(a.shape().size());
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; ++i) {
        if (i != axis) new_shape.push_back(a.shape(i));
    }
    // 单轴内核早于 rank-0 张量出现，把空的归约形状写作 {1}；核心层通过
    // with_reduced_shape 把结果重塑为真正的 0-dim 张量，所以该回退永远不会到达
    // 公开 API。
    if (new_shape.empty()) new_shape = {1};

    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    Tensor result(new_shape, reduction_type(a.dtype()), Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, "Sum");

    // 参数布局恰好与 SoftmaxParams 一致。
    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::SumVec, a.dtype(),
                            {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor ProdImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Prod");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Prod: missing VulkanContext");

    const int dims = static_cast<int>(a.shape().size());
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; ++i) {
        if (i != axis) new_shape.push_back(a.shape(i));
    }
    // 单轴内核早于 rank-0 张量出现，把空的归约形状写作 {1}；核心层通过
    // with_reduced_shape 把结果重塑为真正的 0-dim 张量，所以该回退永远不会到达
    // 公开 API。
    if (new_shape.empty()) new_shape = {1};

    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    Tensor result(new_shape, reduction_type(a.dtype()), Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, "Prod");

    // 参数布局恰好与 SoftmaxParams 一致。
    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::ProdVec, a.dtype(),
                            {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor MeanImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Mean");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Mean: missing VulkanContext");

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < a.shape().size(); ++i) {
        if (static_cast<int>(i) != axis) new_shape.push_back(a.shape(i));
    }
    if (new_shape.empty()) new_shape = {1};

    const int dims    = static_cast<int>(a.shape().size());
    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    Tensor result(new_shape, float_reduction_type(a.dtype()), Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(res_impl, "Mean");
    ctx_impl->submitCompute(OpType::Mean, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor MinImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Min");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Min: missing VulkanContext");

    const int dims = static_cast<int>(a.shape().size());
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; ++i) {
        if (i != axis) new_shape.push_back(a.shape(i));
    }
    if (new_shape.empty()) new_shape = {1};

    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    Tensor result(new_shape, a.dtype(), Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, "Min");

    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::MinVec, a.dtype(),
                            {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor MaxImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Max");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Max: missing VulkanContext");

    const int dims = static_cast<int>(a.shape().size());
    std::vector<int64_t> new_shape;
    for (int i = 0; i < dims; ++i) {
        if (i != axis) new_shape.push_back(a.shape(i));
    }
    if (new_shape.empty()) new_shape = {1};

    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    Tensor result(new_shape, a.dtype(), Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, "Max");

    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::MaxVec, a.dtype(),
                            {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor ArgMaxImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Argmax");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Argmax: missing VulkanContext");

    const auto a_shape  = a.shape();
    const int dim       = static_cast<int>(a_shape.size());
    int32_t outer_dim   = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a_shape[i]);
    const int32_t axis_dim = static_cast<int32_t>(a_shape[axis]);
    int32_t inner_dim      = 1;
    for (int i = axis + 1; i < dim; ++i) inner_dim *= static_cast<int32_t>(a_shape[i]);

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < a.shape().size(); ++i) {
        if (static_cast<int>(i) != axis) new_shape.push_back(a.shape(i));
    }
    if (new_shape.empty()) new_shape = {1};

    Tensor result(new_shape, DataType::INT32, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(res_impl, "Argmax");

    SoftmaxParams params{
        .axis_dim = axis_dim,
        // 着色器用它做边界检查，而 dst 只有 outer_dim*inner_dim 个元素；用 numel
        // 会让越界的线程写过 buffer 末尾。
        .outer_dim = outer_dim * inner_dim,
        .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::Argmax, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

Tensor ArgMinImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Argmin");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Argmin: missing VulkanContext");

    const auto a_shape  = a.shape();
    const int dim       = static_cast<int>(a_shape.size());
    int32_t outer_dim   = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a_shape[i]);
    const int32_t axis_dim = static_cast<int32_t>(a_shape[axis]);
    int32_t inner_dim      = 1;
    for (int i = axis + 1; i < dim; ++i) inner_dim *= static_cast<int32_t>(a_shape[i]);

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < a.shape().size(); ++i) {
        if (static_cast<int>(i) != axis) new_shape.push_back(a.shape(i));
    }
    if (new_shape.empty()) new_shape = {1};

    Tensor result(new_shape, DataType::INT32, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(res_impl, "Argmin");

    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::Argmin, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

bool AnyImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Any");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Any: missing VulkanContext");

    // bool 无法传给着色器，所以用 INT32 张量；着色器的原子操作不支持 int8。
    auto result = Tensor::Fill({1}, 0.0f, DataType::INT32, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(res_impl, "Any");

    const int64_t numel = a.numel();
    ValueParams<float> params{.value = val, .numel = numel};

    ctx_impl->submitCompute(OpType::Any, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>((numel + 255) / 256), 1, 1, &params,
                            sizeof(params));

    result.to_host();
    return static_cast<int*>(result.data())[0] == 1;
}

bool AllImpl<Device::VULKAN>::execute(const Tensor& a, float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "All");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan All: missing VulkanContext");

    auto result = Tensor::Fill({1}, 1.0f, DataType::INT32, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(res_impl, "All");

    const int64_t numel = a.numel();
    ValueParams<float> params{.value = val, .numel = numel};

    ctx_impl->submitCompute(OpType::All, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>((numel + 255) / 256), 1, 1, &params,
                            sizeof(params));

    result.to_host();
    return static_cast<int*>(result.data())[0] == 1;
}

}  // namespace ops
