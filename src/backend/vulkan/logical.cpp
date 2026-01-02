#include "backend/vulkan/ops/logical.h"
#include "backend/vulkan/ops/reduce.h"
using namespace via;

namespace ops {
Tensor EqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::Equal,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );

    return res;
}

Tensor NotEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::NotEqual,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );

    return res;
}
Tensor GreaterImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::Greater,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );

    return res;
}
Tensor LessImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::Less,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return res;
}
Tensor GreaterEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());

    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::GreaterEqual,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );
    return res;
}
Tensor LessEqualImpl<Device::VULKAN>::execute(const Tensor& a,const Tensor& b) {
    auto a_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());

    auto b_impl =  std::dynamic_pointer_cast<VKTensor>(b.get_impl());

    auto A = data_as_const_variant(a.dtype(),a.data());
    auto B = data_as_const_variant(b.dtype(),b.data());
    Tensor res(a.shape(), DataType::INT8, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();

    ctx_impl->submitCompute(
        OpType::LessEqual,
        a.dtype(),
        {a_impl->buffer(), b_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );

    return res;
}
size_t NonZeroImpl<Device::VULKAN>::execute(const Tensor& a) {
    auto a_impl = std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(a_impl->context());
    Tensor res = Tensor::Fill({256}, 0, DataType::INT32);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    int64_t numel = a.numel();
    ctx_impl->submitCompute(
        OpType::Nonzero,
        a.dtype(),
        {a_impl->buffer(), res_impl->buffer()},
        (numel + 255) / 256, 1, 1,
        &numel,
        sizeof(int64_t)
    );

    res.to_host();
    size_t count = 0;
    auto res_data = static_cast<int32_t*>(res.data());
    for (int i = 0; i < 256; ++i) {
        count += res_data[i];
    }
    return count;
}

template struct EqualImpl<Device::VULKAN>;
template struct NotEqualImpl<Device::VULKAN>;
template struct GreaterImpl<Device::VULKAN>;
template struct LessImpl<Device::VULKAN>;
template struct GreaterEqualImpl<Device::VULKAN>;
template struct LessEqualImpl<Device::VULKAN>;
template struct NonZeroImpl<Device::VULKAN>;
}