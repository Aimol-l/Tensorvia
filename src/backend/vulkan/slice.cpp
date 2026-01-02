
#include "backend/vulkan/ops/slice.h"
using namespace via;

namespace ops {

Tensor SliceImpl<Device::VULKAN>::execute(
    const Tensor& t,
    const std::vector<std::pair<int64_t, int64_t>>& ranges)
{
    const int ndim = t.shape().size();
    const int slice_dims = ranges.size();

    if (slice_dims > ndim)
        throw std::runtime_error("Slice: invalid ranges");

    // ---------- output shape ----------
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim);
    for (int i = 0; i < slice_dims; ++i)
        out_shape.push_back(ranges[i].second - ranges[i].first);
    for (int i = slice_dims; i < ndim; ++i)
        out_shape.push_back(t.shape(i));

    Tensor res(out_shape, t.dtype(), Device::VULKAN);

    auto res_impl = std::dynamic_pointer_cast<VKTensor>(res.get_impl());
    auto src_impl = std::dynamic_pointer_cast<VKTensor>(t.get_impl());
    auto ctx = std::dynamic_pointer_cast<VulkanContext>(res_impl->context());

    // ---------- fill params ----------
    SliceParams p{};
    p.ndim = ndim;
    p.slice_dims = slice_dims;

    auto in_shape   = t.shape();
    auto out_shape2 = res.shape();
    auto in_strides = t.strides();

    for (int i = 0; i < 6; ++i) {
        p.input_shape[i]   = (i < in_shape.size())   ? in_shape[i]   : 1;
        p.output_shape[i]  = (i < out_shape2.size()) ? out_shape2[i] : 1;
        p.input_strides[i] = (i < in_strides.size()) ? in_strides[i] : 0;
        p.slice_starts[i]  = (i < slice_dims) ? ranges[i].first : 0;
    }

    uint32_t gx = (res.numel() + 255) / 256;
    ctx->submitCompute(
        OpType::Slice,
        t.dtype(),
        {src_impl->buffer(),res_impl->buffer()},
        gx, 1, 1,
        &p, sizeof(p)
    );

    return res;
}

template struct SliceImpl<Device::VULKAN>;

} // namespace ops
