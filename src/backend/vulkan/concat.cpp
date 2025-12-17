#include <stdexcept>
#include <vector>
#include "ops.h"
#include "backend/vulkan/ops/concat.h"

namespace ops {

Tensor ConcatImpl<Device::VULKAN>::execute(
    const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("Concat: empty inputs");
    if (tensors.size() > 8)
        throw std::runtime_error("Concat: max 8 inputs");

    const int ndim = tensors[0].shape().size();
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error("Concat: invalid axis");
    DataType dtype = tensors[0].dtype();
    // ---------- output shape ----------
    std::vector<int64_t> out_shape = tensors[0].shape();
    out_shape[dim] = 0;
    for (auto& t : tensors){
        out_shape[dim] += t.shape(dim);
    }
    Tensor res(out_shape, dtype, Device::VULKAN);
    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    auto ctx =  std::dynamic_pointer_cast<VulkanContext>(res_impl->context());
    // ---------- fill params ----------
    ConcatParams p{};
    p.num  = tensors.size();
    p.axis = dim;
    p.prefix_sum[0] = 0;
    for (int i = 0; i < tensors.size(); ++i){
        p.prefix_sum[i + 1] = p.prefix_sum[i] + tensors[i].shape(dim);
    }
    auto out_strides = res.strides();
    for (int i = 0; i < 6; ++i) {
        p.output_shape[i]  = (i < out_shape.size()) ? out_shape[i] : 1;
        p.output_strides[i] = (i < out_strides.size()) ? out_strides[i] : 0;
    }
    for (int i = 0; i < tensors.size(); ++i) {
        auto in_strides = tensors[i].strides();
        for (int j = 0; j < 6; ++j){
            p.input_strides[i][j] = (j < in_strides.size()) ? in_strides[j] : 0;
        }
    }

    auto params_buf = ctx->createBuffer<ConcatParams>(p);
    // ---------- collect input buffers ----------
    std::vector<vk::Buffer> buffers;
    buffers.push_back(params_buf);
    buffers.push_back(res_impl->buffer());
    for (int i = 0; i < 8; ++i) {
        if (i < tensors.size()) {
            auto impl = std::dynamic_pointer_cast<VKTensor>(tensors[i].get_impl());
            buffers.push_back(impl->buffer());
        }else{
            auto impl = std::dynamic_pointer_cast<VKTensor>(tensors[0].get_impl());
            buffers.push_back(impl->buffer());
        }
    }
    uint32_t gx = (res.numel() + 255) / 256;
    ctx->submitCompute(
        OpType::Concat,
        dtype,
        buffers,
        gx, 1, 1,
        nullptr, 0
    );
    return res;
}


template struct ConcatImpl<Device::VULKAN>;

} // namespace ops
