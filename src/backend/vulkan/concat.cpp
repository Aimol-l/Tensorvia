#include <limits>
#include "backend/vulkan/ops/concat.h"

namespace ops {

// 1.先把n个张量按照顺序复制到一个大的temp tensor里面，同时记录offsets[n]。
// 2.把temp tensor concat 到 res tensor 里面。
Tensor ConcatImpl<Device::VULKAN>::execute(const std::vector<Tensor> &tensors, int dim) {
    // 最大支持8个张量拼接
    if(tensors.size() > 8){
        throw std::runtime_error("Vulkan Concat only support max 8 tensors!");
    }
    DataType dtype = tensors[0].dtype();
    // 2. 计算输出张量的形状
    std::vector<int64_t> output_shape = tensors[0].shape();
    output_shape[dim] = 0;
    for (const auto& t : tensors) {
        output_shape[dim] += t.shape()[dim];
    }
    // 3. 创建输出张量
    Tensor tmp(output_shape, dtype, Device::VULKAN);
    Tensor res(output_shape, dtype, Device::VULKAN);

    // 4. 获取VULKAN队列
    auto temp_impl = std::dynamic_pointer_cast<VKTensor>(tmp.get_impl());
    auto dst_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());

    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(temp_impl->context());

    // 5.先把所有的tensor复制到tmp里面，同时记录offsets
    std::vector<uint32_t> offsets(tensors.size(),0);
    for(int i = 1;i<tensors.size();i++){
        offsets[i] = uint32_t(tensors[i-1].numel()) + offsets[i-1];
    }
    for(int i = 0;i<tensors.size();i++){
        CopyParams params{
            .subnumel = uint32_t(tensors[i].numel()),
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
            {temp_impl->buffer(),src_impl->buffer()},
            gx, 1, 1,
            &params,
            sizeof(CopyParams)
        );
    }
    // 已经把所有子张量复制到目标张量，可以开始concat了
    // 6. 准备Concat参数
    ConcatParams concat_params;
    concat_params.num = static_cast<uint32_t>(tensors.size());
    concat_params.axis = static_cast<uint32_t>(dim);
    for(int i = 0;i<tensors.size();i++){
        concat_params.offsets[i] = static_cast<uint32_t>(offsets[i]);
        concat_params.input_sizes[i] = static_cast<uint32_t>(tensors[i].shape(dim));
        // 获取输入张量的stride
        auto in_strides = tensors[i].strides();
        for(int j =0;j<in_strides.size();j++){
            concat_params.input_strides[i][j] = static_cast<uint32_t>(in_strides[j]);
        }
    }
    for(int i =0;i<output_shape.size();i++){
        concat_params.output_strides[i] = static_cast<uint32_t>(res.strides(i));
    }
    auto params_buffer = ctx_impl->createBuffer<ConcatParams>(concat_params);
    // 7. 调用Vulkan计算管线执行concat
    auto tmp_impl = std::dynamic_pointer_cast<VKTensor>(tmp.get_impl());
    ctx_impl->submitCompute(    
        OpType::Concat,
        dtype,
        {params_buffer, tmp_impl->buffer(), dst_impl->buffer()},
        (res.numel() + 255) / 256, 1, 1,
        nullptr,0
    );
    return res;
}

template struct ConcatImpl<Device::VULKAN>;
}