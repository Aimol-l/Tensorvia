
#include "backend/vulkan/ops/transpose.h"

namespace ops {

void TransposeImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    std::vector<int64_t> shape = {a.shape(1),a.shape(0)};
    a.reshape(shape);
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
Tensor TransposeImpl<Device::VULKAN>::execute(const Tensor& a,std::initializer_list<int64_t> axes){
    // 创建结果张量
    std::vector<int64_t> new_shape;
    std::vector<int64_t> axes_v(axes);
    for(auto axe:axes)  new_shape.push_back(a.shape(axe));
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    return result;
}
template struct TransposeImpl<Device::VULKAN>;
}  // namespace ops