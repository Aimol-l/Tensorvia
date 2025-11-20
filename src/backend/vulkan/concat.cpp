#include <limits>
#include "backend/vulkan/ops/concat.h"

namespace ops {

Tensor ConcatImpl<Device::VULKAN>::execute(const std::vector<Tensor> &tensors, int dim) {
    const auto& first_shape = tensors[0].shape();
    DataType dtype = tensors[0].dtype();
    Device device = tensors[0].device();
    // 2. 计算输出张量的形状
    std::vector<int64_t> output_shape = first_shape;
    output_shape[dim] = 0;
    for (const auto& t : tensors) {
        output_shape[dim] += t.shape()[dim];
    }
    // 3. 创建输出张量
    Tensor output(output_shape, dtype, device);
    // 4. 获取VULKAN队列
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(tensors[0].get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    return output;
}

template struct ConcatImpl<Device::VULKAN>;
}