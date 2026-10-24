#include <cstdint>
#include <memory>
#include <stdexcept>
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

void TransposeImpl<Device::VULKAN>::execute(Tensor& a) {
    // 2D-only transpose.
    const int rows = static_cast<int>(a.shape(0));
    const int cols = static_cast<int>(a.shape(1));
    Tensor output({cols, rows}, a.dtype(), a.device());
    constexpr int TILE_SZ = 32;
    const uint32_t gx = static_cast<uint32_t>((cols + TILE_SZ - 1) / TILE_SZ);
    const uint32_t gy = static_cast<uint32_t>((rows + TILE_SZ - 1) / TILE_SZ);

    auto src_ptr = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_ptr = std::dynamic_pointer_cast<VKTensor>(output.get_impl());
    if (!src_ptr || !dst_ptr) throw std::runtime_error("vulkan Transpose: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_ptr->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Transpose: missing VulkanContext");

    Trans2DParams params{.rows = static_cast<uint32_t>(rows),
                         .cols = static_cast<uint32_t>(cols)};
    ctx_impl->submitCompute(OpType::Transpose2d, a.dtype(),
                            {src_ptr->buffer(), dst_ptr->buffer()}, gx, gy, 1, &params,
                            sizeof(params));
    a = std::move(output);
}

namespace {

// Shared by the out-of-place and in-place N-D transposes.
TransNDParams make_transpose_params(const Tensor& a, const Tensor& dst,
                                    const std::vector<int64_t>& new_shape,
                                    const std::vector<int32_t>& axes_norm) {
    const int ndim = static_cast<int>(a.shape().size());
    std::vector<int32_t> in_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i)
        in_strides[i] = in_strides[i + 1] * static_cast<int32_t>(a.shape(i + 1));

    std::vector<int32_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i)
        out_strides[i] = out_strides[i + 1] * static_cast<int32_t>(dst.shape(i + 1));

    TransNDParams params{};
    params.numel = static_cast<int32_t>(a.numel());
    params.dims  = ndim;
    for (int i = 0; i < ndim; ++i) {
        params.axes[i]        = axes_norm[i];
        params.in_strides[i]  = in_strides[i];
        params.out_strides[i] = out_strides[i];
    }
    (void)new_shape;
    return params;
}

std::vector<int32_t> normalize_axes(std::initializer_list<int64_t> axes, int32_t ndim) {
    std::vector<int32_t> axes_norm;
    std::vector<bool> seen(static_cast<size_t>(ndim), false);
    axes_norm.reserve(axes.size());
    for (int64_t ax : axes) {
        int64_t ax_norm = ax;
        if (ax_norm < 0) ax_norm += ndim;
        if (ax_norm < 0 || ax_norm >= ndim || seen[static_cast<size_t>(ax_norm)])
            throw std::invalid_argument("invalid axes");
        seen[static_cast<size_t>(ax_norm)] = true;
        axes_norm.push_back(static_cast<int32_t>(ax_norm));
    }
    return axes_norm;
}

void submit_transpose(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes) {
    const int32_t ndim = static_cast<int32_t>(a.shape().size());
    if (axes.size() != static_cast<size_t>(ndim))
        throw std::invalid_argument("axes size must equal tensor ndim");

    std::vector<int64_t> new_shape;
    for (int64_t ax : axes) new_shape.push_back(a.shape(ax < 0 ? ax + ndim : ax));
    const auto axes_norm = normalize_axes(axes, ndim);

    const TransNDParams params = make_transpose_params(a, dst, new_shape, axes_norm);

    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    if (!src_impl || !dst_impl)
        throw std::runtime_error("vulkan Transpose: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan Transpose: missing VulkanContext");

    const uint32_t gx = static_cast<uint32_t>((params.numel + 255) / 256);
    ctx_impl->submitCompute(OpType::TransposeNd, a.dtype(),
                            {src_impl->buffer(), dst_impl->buffer()}, gx, 1, 1, &params,
                            sizeof(params));
}

}  // namespace

void TransposeImpl<Device::VULKAN>::execute(const Tensor& a, Tensor& dst,
                                            std::initializer_list<int64_t> axes) {
    submit_transpose(a, dst, axes);
}

Tensor TransposeImpl<Device::VULKAN>::execute(const Tensor& a,
                                              std::initializer_list<int64_t> axes) {
    const int32_t ndim = static_cast<int32_t>(a.shape().size());
    if (axes.size() != static_cast<size_t>(ndim))
        throw std::invalid_argument("axes size must equal tensor ndim");

    std::vector<int64_t> new_shape;
    for (int64_t ax : axes) new_shape.push_back(a.shape(ax < 0 ? ax + ndim : ax));

    Tensor result(new_shape, a.dtype(), Device::VULKAN);
    submit_transpose(a, result, axes);
    return result;
}

}  // namespace ops
