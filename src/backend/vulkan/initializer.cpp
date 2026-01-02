#include "backend/vulkan/ops/initializer.h"

using namespace via;

namespace ops {
Tensor ZerosImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype){
   Tensor temp(shape, dtype, Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(temp.get_impl());
    if (!src_impl) {
        throw std::runtime_error("Failed to cast to VKTensor");
    }
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (ctx_impl.get() == nullptr) {
        throw std::runtime_error("VulkanContext is null!");
    }

    ValueParams<float> params{
        .value = 0,
        .numel = static_cast<int64_t>(temp.numel())
    };
    ctx_impl->submitCompute(
        OpType::Fill, 
        temp.dtype(),
        {src_impl->buffer()},
        (temp.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
    return  temp;
}

Tensor OnesImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype){
    Tensor temp(shape, dtype, Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(temp.get_impl());
    if (!src_impl) {
        throw std::runtime_error("Failed to cast to VKTensor");
    }
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (ctx_impl.get() == nullptr) {
        throw std::runtime_error("VulkanContext is null!");
    }

    ValueParams<float> params{
        .value = 1,
        .numel = static_cast<int64_t>(temp.numel())
    };
    ctx_impl->submitCompute(
        OpType::Fill, 
        temp.dtype(),
        {src_impl->buffer()},
        (temp.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
    return  temp;
}

Tensor FillImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype, float value){
    Tensor temp(shape, dtype, Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(temp.get_impl());
    if (!src_impl) {
        throw std::runtime_error("Failed to cast to VKTensor");
    }
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (ctx_impl.get() == nullptr) {
        throw std::runtime_error("VulkanContext is null!");
    }

    ValueParams<float> params{
        .value = value,
        .numel = static_cast<int64_t>(temp.numel())
    };
    ctx_impl->submitCompute(
        OpType::Fill, 
        temp.dtype(),
        {src_impl->buffer()},
        (temp.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
    return  temp;
}

Tensor RandomImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype,float min,float max){
    Tensor temp(shape, dtype, Device::VULKAN);
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(temp.get_impl());
    if (!src_impl) {
        throw std::runtime_error("Failed to cast to VKTensor");
    }
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (ctx_impl.get() == nullptr) {
        throw std::runtime_error("VulkanContext is null!");
    }
    std::random_device rd;
    uint32_t seed =   rd();
    RandomParams params{
        .min = min,
        .max = max,
        .seed = seed,
        .numel = static_cast<int64_t>(temp.numel())
    };
    ctx_impl->submitCompute(
        OpType::Random, 
        temp.dtype(),
        {src_impl->buffer()},
        (temp.numel() + 255) / 256, 1, 1,
        &params, 
        sizeof(params)
    );
    return  temp;
}

template struct ZerosImpl<Device::VULKAN>;
template struct OnesImpl<Device::VULKAN>;
template struct FillImpl<Device::VULKAN>;
template struct RandomImpl<Device::VULKAN>;

}