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

constexpr int32_t kFlatReduceBlocks   = 1024;
constexpr int64_t kFlatReduceMinNumel = 1 << 18;

Tensor axis_reduce_vk(const Tensor& a, int axis, DataType out_type, OpType op,
                      const char* what) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, what);
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");

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

    Tensor result(new_shape, out_type, Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, what);

    if (outer_dim * inner_dim == 1 && axis_dim >= kFlatReduceMinNumel) {
        const int32_t chunk = static_cast<int32_t>((static_cast<int64_t>(axis_dim) +
                                                    kFlatReduceBlocks - 1) /
                                                   kFlatReduceBlocks);
        Tensor partial({kFlatReduceBlocks}, out_type, Device::VULKAN);
        auto partial_impl = std::dynamic_pointer_cast<VKTensor>(partial.get_impl());
        check_vk(partial_impl, what);

        // 趟 1：Mean 先算纯部分和——末段短于 chunk，先除会少算，分母留到趟 2。
        // numel 传真实元素数：chunk 是 ceil 出来的，B*chunk 会比它大，末段靠这个
        // 字段截断。
        const OpType pass1_op = (op == OpType::Mean) ? OpType::SumVec : op;
        const ReduceVecParams first{.axis_dim = chunk,
                                    .outer_dim = kFlatReduceBlocks,
                                    .inner_dim = 1,
                                    .numel = axis_dim,
                                    .mean_n = 0};
        ctx_impl->submitCompute(pass1_op, a.dtype(),
                                {src_impl->buffer(), partial_impl->buffer()},
                                static_cast<uint32_t>((kFlatReduceBlocks + 255) / 256), 1, 1,
                                &first, sizeof(ReduceVecParams));

        // 趟 2：折叠 B 个部分值。输入就是 partial，元素数恰好 B。Mean 的分母是
        // 全归约的元素数，不是趟 2 自己的 axis_dim（=B）。
        const ReduceVecParams second{.axis_dim = kFlatReduceBlocks,
                                     .outer_dim = 1,
                                     .inner_dim = 1,
                                     .numel = kFlatReduceBlocks,
                                     .mean_n = (op == OpType::Mean) ? axis_dim : 0};
        ctx_impl->submitCompute(op, out_type, {partial_impl->buffer(), result_impl->buffer()},
                                1, 1, 1, &second, sizeof(ReduceVecParams));
        return result;
    }

    // 参数布局与 SoftmaxParams 一致，外加 numel 与 mean_n。
    const ReduceVecParams params{.axis_dim = axis_dim,
                                 .outer_dim = outer_dim * inner_dim,
                                 .inner_dim = inner_dim,
                                 .numel = static_cast<int32_t>(a.numel()),
                                 .mean_n = 0};
    ctx_impl->submitCompute(op, a.dtype(), {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(ReduceVecParams));
    return result;
}

}  // namespace

Tensor SumImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    // Sum 把窄整型与 16 位浮点拓宽到真正的累加类型；所有后端遵循同一条
    // reduction_type 规则。
    return axis_reduce_vk(a, axis, reduction_type(a.dtype()), OpType::SumVec, "Sum");
}

Tensor ProdImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    // Prod 与 Sum 同族：窄整型与 16 位浮点拓宽到 reduction_type 再累加。
    return axis_reduce_vk(a, axis, reduction_type(a.dtype()), OpType::ProdVec, "Prod");
}

Tensor MeanImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    return axis_reduce_vk(a, axis, float_reduction_type(a.dtype()), OpType::Mean, "Mean");
}

Tensor MinImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    return axis_reduce_vk(a, axis, a.dtype(), OpType::MinVec, "Min");
}

Tensor MaxImpl<Device::VULKAN>::execute(const Tensor& a, int axis) {
    return axis_reduce_vk(a, axis, a.dtype(), OpType::MaxVec, "Max");
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
