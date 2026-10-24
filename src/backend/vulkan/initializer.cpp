#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

namespace {
template <class Params>
void fill_like(const Tensor& temp, const char* what, OpType op, const Params& params) {
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(temp.get_impl());
    if (!src_impl) throw std::runtime_error(std::string("vulkan ") + what + ": expected a VKTensor");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());
    if (!ctx_impl)
        throw std::runtime_error(std::string("vulkan ") + what + ": missing VulkanContext");
    ctx_impl->submitCompute(op, temp.dtype(), {src_impl->buffer()},
                            static_cast<uint32_t>((temp.numel() + 255) / 256), 1, 1, &params,
                            sizeof(params));
}
}  // namespace

Tensor ZerosImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::VULKAN);
    ValueParams<float> params{.value = 0.0f, .numel = static_cast<int64_t>(temp.numel())};
    fill_like(temp, "Zeros", OpType::Fill, params);
    return temp;
}

Tensor OnesImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::VULKAN);
    ValueParams<float> params{.value = 1.0f, .numel = static_cast<int64_t>(temp.numel())};
    fill_like(temp, "Ones", OpType::Fill, params);
    return temp;
}

Tensor FillImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                         float value) {
    Tensor temp(shape, dtype, Device::VULKAN);
    ValueParams<float> params{.value = value, .numel = static_cast<int64_t>(temp.numel())};
    fill_like(temp, "Fill", OpType::Fill, params);
    return temp;
}

void FillImpl<Device::VULKAN>::execute(Tensor& a, float value) {
    ValueParams<float> params{.value = value, .numel = static_cast<int64_t>(a.numel())};
    fill_like(a, "Fill", OpType::Fill, params);
}

Tensor RandomImpl<Device::VULKAN>::execute(const std::vector<int64_t>& shape, DataType dtype,
                                           float min, float max) {
    Tensor temp(shape, dtype, Device::VULKAN);
    std::random_device rd;
    RandomParams params{
        .min = min, .max = max, .seed = rd(), .numel = static_cast<int64_t>(temp.numel())};
    fill_like(temp, "Random", OpType::Random, params);
    return temp;
}

}  // namespace ops
