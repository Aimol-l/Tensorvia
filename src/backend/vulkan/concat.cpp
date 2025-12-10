#include <limits>
#include "backend/vulkan/ops/concat.h"

namespace ops {

Tensor ConcatImpl<Device::VULKAN>::execute(const std::vector<Tensor> &tensors, int dim) {
    // 最大支持8个张量拼接
    if(tensors.size() > 8){
        throw std::runtime_error("Vulkan Concat only support max 8 tensors!");
    }
    const auto& first_shape = tensors[0].shape();
    DataType dtype = tensors[0].dtype();
    // 2. 计算输出张量的形状
    std::vector<int64_t> output_shape = first_shape;
    output_shape[dim] = 0;
    for (const auto& t : tensors) {
        output_shape[dim] += t.shape()[dim];
    }
    // 3. 创建输出张量
    // Tensor tmp(output_shape, dtype, Device::VULKAN);
    Tensor res(output_shape, dtype, Device::VULKAN);

    // 4. 获取VULKAN队列
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(dst_impl->context());

    // 5.先把所有的tensor复制到tmp里面，同时记录offsets
    std::vector<int64_t> offsets(tensors.size(),0);
    for(int i = 1;i<tensors.size();i++){
        offsets[i] = tensors[i-1].numel() + offsets[i-1];
    }
    for(int i = 0;i<tensors.size();i++){
        CopyParams params{
            .subnumel = tensors[i].numel(),
            .offset = offsets[i]
        };
        auto src_impl = std::dynamic_pointer_cast<VKTensor>(tensors[i].get_impl());
        constexpr int THREADS = 128;
        constexpr int BLOCK_SIZE = 64;
        // ceil(10000 / (64*128)) = 2
        uint32_t gx = std::ceil(tensors[i].numel() / float(BLOCK_SIZE * THREADS));
        ctx_impl->submitCompute(
            OpType::ConcatAdd,
            dtype,
            {dst_impl->buffer(),src_impl->buffer()},
            gx, 1, 1,
            &params,
            sizeof(CopyParams)
        );
    }
    // 已经把所有子张量复制到目标张量，可以开始concat了
    
    return res;
}

template struct ConcatImpl<Device::VULKAN>;
}