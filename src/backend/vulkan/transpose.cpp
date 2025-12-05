
#include "backend/vulkan/ops/transpose.h"

namespace ops {

void TransposeImpl<Device::VULKAN>::execute(Tensor& a){
    // 这个只支持二维转置
    const int rows = a.shape(0);
    const int cols = a.shape(1);
    Tensor output({cols, rows}, a.dtype(), a.device());
    // 分块尺寸
    constexpr int TILE_SZ = 32;
    uint32_t gx = (cols + TILE_SZ - 1) / TILE_SZ;
    uint32_t gy = (rows + TILE_SZ - 1) / TILE_SZ;
    uint32_t gz = 1;
    auto src_ptr = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_ptr = std::dynamic_pointer_cast<VKTensor>(output.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_ptr->context());
    Trans2DParams params{
        .rows = uint32_t(rows),
        .cols = uint32_t(cols)
    };
    ctx_impl->submitCompute(
        OpType::Transpose2d,
        a.dtype(),
        {src_ptr->buffer(),dst_ptr->buffer()},
        gx,gy,gz,
        &params,
        sizeof(params)
    );
    a = std::move(output);
}
void TransposeImpl<Device::VULKAN>::execute(const Tensor&a,Tensor& dst,std::initializer_list<int64_t> axes){
    // 创建结果张量
    std::vector<int64_t> new_shape;
    std::vector<int64_t> axes_v(axes);
    for(auto axe:axes)  new_shape.push_back(a.shape(axe));
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor TransposeImpl<Device::VULKAN>::execute(const Tensor& a, std::initializer_list<int64_t> axes) {
    const int32_t ndim = static_cast<int32_t>(a.shape().size());
    if (axes.size() != ndim) {
        throw std::invalid_argument("axes size must equal tensor ndim");
    }

    // 构建 new_shape 并标准化 axes
    std::vector<int64_t> new_shape;
    std::vector<int32_t> axes_norm;
    std::vector<bool> seen(ndim, false);
    for (int64_t ax : axes) {
        int64_t ax_norm = ax;
        if (ax_norm < 0) ax_norm += ndim;
        if (ax_norm < 0 || ax_norm >= ndim || seen[ax_norm]) {
            throw std::invalid_argument("invalid axes");
        }
        seen[ax_norm] = true;
        axes_norm.push_back(static_cast<int32_t>(ax_norm));
        new_shape.push_back(a.shape(ax_norm));
    }

    Tensor result(new_shape, a.dtype(), Device::VULKAN);

    // 计算 input strides (row-major)
    std::vector<int32_t> in_strides(ndim, 1);
    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * static_cast<int32_t>(a.shape(i + 1));
    }

    // 计算 output strides (row-major)
    std::vector<int32_t> out_strides(ndim, 1);
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * static_cast<int32_t>(result.shape(i + 1));
    }

    // 准备 push constant
    TransNDParams params{};
    params.numel = static_cast<int32_t>(a.numel());
    params.dims = ndim;
    for (int i = 0; i < ndim; ++i) {
        params.axes[i] = axes_norm[i];
        params.in_strides[i] = in_strides[i];
        params.out_strides[i] = out_strides[i];
    }

    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    uint32_t gx = (params.numel + 255) / 256;
    ctx_impl->submitCompute(
        OpType::TransposeNd,
        a.dtype(),
        {src_impl->buffer(), dst_impl->buffer()},
        gx, 1, 1,
        &params,
        sizeof(params)
    );

    return result;
}

template struct TransposeImpl<Device::VULKAN>;
}  // namespace ops