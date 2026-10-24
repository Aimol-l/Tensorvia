#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/vulkan/vulkan_ops.hpp"
#include "interface/backend/vulkan/vulkan_common.hpp"
#include "interface/backend/vulkan/vulkan_context.hpp"
#include "interface/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {

// NaN-propagating fold contracts: a plain `v < a ? v : a` swallows NaN.
inline bool contract_is_nan(double v) { return v != v; }
inline double min_contract(double a, double b) {
    if (contract_is_nan(a)) return a;
    if (contract_is_nan(b)) return b;
    return b < a ? b : a;
}
inline double max_contract(double a, double b) {
    if (contract_is_nan(a)) return a;
    if (contract_is_nan(b)) return b;
    return b > a ? b : a;
}

struct FoldOp {
    double init;
    double (*apply)(double, double);
};

/// Folds the per-workgroup partials on the host.
///
/// The scratch tensor has one slot per workgroup (the shader writes
/// dst.data[gl_WorkGroupID.x]), so its size is (numel+255)/256, not a fixed
/// 256: a larger block count would write out of bounds and a smaller one would
/// read never-written slots. Reads use the scratch dtype because the scratch
/// uses the promoted accumulator dtype, not a.dtype().
double fold_partials(Tensor& scratch, int64_t blocks, FoldOp op) {
    scratch.to_host();
    double acc = op.init;
    dispatch_dtype(scratch.dtype(), [&](auto type_id) {
        using T       = typename decltype(type_id)::type;
        const T* p    = static_cast<const T*>(scratch.data());
        for (int64_t i = 0; i < blocks; ++i) acc = op.apply(acc, static_cast<double>(p[i]));
    });
    return acc;
}

inline void check_vk(const std::shared_ptr<TensorImpl>& impl, const char* what) {
    if (!impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
}

}  // namespace

float SumImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Sum");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Sum: missing VulkanContext");

    const int64_t numel  = a.numel();
    const int64_t blocks = (numel + 255) / 256;
    // Scratch uses the accumulator dtype: the shader promotes block-local sums
    // (int8/int16 -> int32, int32/int64 -> int64, f16/bf16 -> float).
    Tensor res({blocks > 0 ? blocks : 1}, reduction_type(a.dtype()), Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    check_vk(res_impl, "Sum");

    ctx_impl->submitCompute(OpType::Sum, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(blocks), 1, 1, &numel, sizeof(int64_t));
    return static_cast<float>(
        fold_partials(res, blocks, {0.0, [](double x, double v) { return x + v; }}));
}

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
    // Removing the only dimension of a 1D input leaves an empty shape; tensors
    // do not support rank 0, so fall back to {1} (matching the CPU side).
    if (new_shape.empty()) new_shape = {1};

    int32_t inner_dim = 1;
    for (int i = axis + 1; i < dims; ++i) inner_dim *= static_cast<int32_t>(a.shape(i));
    const int32_t axis_dim = static_cast<int32_t>(a.shape(axis));
    int32_t outer_dim      = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= static_cast<int32_t>(a.shape(i));

    Tensor result(new_shape, reduction_type(a.dtype()), Device::VULKAN);
    auto result_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    check_vk(result_impl, "Sum");

    // The parameter layout happens to match SoftmaxParams.
    SoftmaxParams params{
        .axis_dim = axis_dim, .outer_dim = outer_dim * inner_dim, .inner_dim = inner_dim};

    ctx_impl->submitCompute(OpType::SumVec, a.dtype(),
                            {src_impl->buffer(), result_impl->buffer()},
                            static_cast<uint32_t>(((outer_dim * inner_dim) + 255) / 256), 1, 1,
                            &params, sizeof(SoftmaxParams));
    return result;
}

float MeanImpl<Device::VULKAN>::execute(const Tensor& a) {
    const int64_t numel = a.numel();
    if (numel == 0) return 0.0f;
    // Divide in double: above 2^24 a float divisor is no longer exact.
    return static_cast<float>(SumImpl<Device::VULKAN>::execute(a) /
                              static_cast<double>(numel));
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

float MinImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Min");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Min: missing VulkanContext");

    const int64_t numel  = a.numel();
    const int64_t blocks = (numel + 255) / 256;
    Tensor res({blocks > 0 ? blocks : 1}, a.dtype(), Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    check_vk(res_impl, "Min");

    ctx_impl->submitCompute(OpType::Min, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(blocks), 1, 1, &numel, sizeof(int64_t));
    return static_cast<float>(
        fold_partials(res, blocks, {std::numeric_limits<double>::infinity(), &min_contract}));
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

float MaxImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    check_vk(src_impl, "Max");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Max: missing VulkanContext");

    const int64_t numel  = a.numel();
    const int64_t blocks = (numel + 255) / 256;
    Tensor res({blocks > 0 ? blocks : 1}, a.dtype(), Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    check_vk(res_impl, "Max");

    ctx_impl->submitCompute(OpType::Max, a.dtype(), {src_impl->buffer(), res_impl->buffer()},
                            static_cast<uint32_t>(blocks), 1, 1, &numel, sizeof(int64_t));
    return static_cast<float>(
        fold_partials(res, blocks, {-std::numeric_limits<double>::infinity(), &max_contract}));
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
        // The shader uses this as its bounds check, and dst only holds
        // outer_dim*inner_dim elements; numel would let out-of-range threads
        // write past the buffer.
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

    // bool cannot be passed to the shader, so use an INT32 tensor; the shader's
    // atomic does not support int8.
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
