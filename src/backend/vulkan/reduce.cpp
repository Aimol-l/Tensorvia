#include "backend/vulkan/ops/reduce.h"

namespace ops {

float SumImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    float res = 0.0f;
    return res;
}
Tensor SumImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    return result;
}
float MeanImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    float sum_val = 0.0f;
    return sum_val / a.numel();
}
Tensor MeanImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    return result;
}
float MinImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor MinImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    return result;
}
float MaxImpl<Device::VULKAN>::execute(const Tensor& a){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
}
Tensor MaxImpl<Device::VULKAN>::execute(const Tensor& a,int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,a.dtype(),Device::VULKAN);
    return result;
}

Tensor ArgMaxImpl<Device::VULKAN>::execute(const Tensor &a, int axis){
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    // 移除 a.shape(axis) 所在的轴
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); i++) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape,DataType::INT32,Device::VULKAN);
    return result;
}

Tensor ArgMinImpl<Device::VULKAN>::execute(const Tensor &a, int axis) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    std::vector<int64_t> new_shape;
    for (int i = 0; i < a.shape().size(); ++i) {
        if (i != axis)  new_shape.push_back(a.shape(i));
    }
    Tensor result(new_shape, DataType::INT32, Device::VULKAN);
    return result;
}
bool AnyImpl<Device::VULKAN>::execute(const Tensor& a,float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    return false;
}
bool AllImpl<Device::VULKAN>::execute(const Tensor& a,float val) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    return false;
}

 template struct SumImpl<Device::VULKAN>;
 template struct MeanImpl<Device::VULKAN>;
 template struct MinImpl<Device::VULKAN>;
 template struct MaxImpl<Device::VULKAN>;
 template struct ArgMaxImpl<Device::VULKAN>;
 template struct ArgMinImpl<Device::VULKAN>;
 template struct AnyImpl<Device::VULKAN>;
 template struct AllImpl<Device::VULKAN>;

}