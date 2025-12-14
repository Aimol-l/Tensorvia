
#include "backend/vulkan/ops/slice.h"

namespace ops {
Tensor SliceImpl<Device::VULKAN>::execute(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges){
    // 计算新的shape
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto& [start, end] = ranges[i];
        new_shape.push_back(end - start);
    }
    // 创建新的Tensor
    Tensor res(new_shape, t.dtype(), t.device());
    // 4. 获取VULKAN队列
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());

    return res;
}
template struct SliceImpl<Device::VULKAN>;

}