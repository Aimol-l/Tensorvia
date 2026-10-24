#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_common.hpp"
#include "tensorvia/backend/vulkan/vulkan_context.hpp"
#include "tensorvia/backend/vulkan/vulkan_tensor.hpp"

namespace ops {

// [w,h] @ [h,w] -> [w,w]
// [b,w,h] @ [b,h,w] -> [b,w,w]
Tensor MatMulImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b) {
    const bool batched = a.shape().size() == 3;
    const int batch     = batched ? static_cast<int>(a.shape(0)) : 1;
    const int rows      = batched ? static_cast<int>(a.shape(1)) : static_cast<int>(a.shape(0));
    const int common    = batched ? static_cast<int>(a.shape(2)) : static_cast<int>(a.shape(1));
    const int cols      = batched ? static_cast<int>(b.shape(2)) : static_cast<int>(b.shape(1));

    // 2D input must produce 2D output, matching the other backends.
    std::vector<int64_t> newshape =
        batched ? std::vector<int64_t>{batch, rows, cols}
                : std::vector<int64_t>{rows, cols};
    const DataType res_type = compute_type(a.dtype(), b.dtype());
    constexpr int TILE_SZ   = 16;

    Tensor dst(newshape, res_type, Device::VULKAN);
    const Tensor A = a.dtype() == res_type ? a : ops::Typecast(a, res_type);
    const Tensor B = b.dtype() == res_type ? b : ops::Typecast(b, res_type);

    auto A_impl   = std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl   = std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl = std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    if (!A_impl || !B_impl || !Dst_impl)
        throw std::runtime_error("vulkan MatMul: expected VKTensors");
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    if (!ctx_impl) throw std::runtime_error("vulkan MatMul: missing VulkanContext");

    MatmulParams params{.batch = static_cast<uint32_t>(batch),
                        .N     = static_cast<uint32_t>(cols),
                        .K     = static_cast<uint32_t>(common),
                        .M     = static_cast<uint32_t>(rows)};
    const uint32_t gx = static_cast<uint32_t>((cols + TILE_SZ - 1) / TILE_SZ);
    const uint32_t gy = static_cast<uint32_t>((rows + TILE_SZ - 1) / TILE_SZ);
    const uint32_t gz = static_cast<uint32_t>(batch);
    ctx_impl->submitCompute(OpType::Matmul, res_type,
                            {A_impl->buffer(), B_impl->buffer(), Dst_impl->buffer()}, gx, gy,
                            gz, &params, sizeof(params));
    return dst;
}

}  // namespace ops
