#include "backend/vulkan/ops/temp.h"
using namespace via;

namespace ops {

void TempImpl<Device::VULKAN>::execute(Tensor& a){
    auto src_impl =  std::dynamic_pointer_cast<VKTensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(src_impl->context());


    TempParams params{
        .M = static_cast<uint32_t>(a.shape(0)),
        .N = static_cast<uint32_t>(a.shape(1))
    };

    // cuda:  Block      -> Thread       | Warp(32)
    // vk:    Workgroup  -> Invocation   | Subgroup(32/64)
    // sycl:  work-group -> work-item    | sub-group(32/64)
    // hip:   Block      -> Thread       | Wavefront(64)

    ctx_impl->submitCompute(
        OpType::Temp,
        a.dtype(),
        {src_impl->buffer()},
        1, params.M, 1,
        &params,
        sizeof(TempParams)
    );

}

template struct TempImpl<Device::VULKAN>;

}