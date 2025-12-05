#include "ops.h"
#include "backend/vulkan/ops/mul.h"


namespace ops {
// [w,h] @ [h,w] --> [w,w]
// [b,w,h] @ [b,h,w] --> [b,w,w]
Tensor MulImpl<Device::VULKAN>::execute(const Tensor& a, const Tensor& b){
    int batch =     a.shape().size() == 3?a.shape(0):1;
    int rows =      a.shape().size() == 3?a.shape(1):a.shape(0);
    int common =    a.shape().size() == 3?a.shape(2):a.shape(1);
    int cols =      a.shape().size() == 3?b.shape(2):b.shape(1);
    std::vector<int64_t> newshape = {batch,rows,cols};
    DataType res_type = compute_type(a.dtype(),b.dtype());
    constexpr int TILE_SZ = 16; // or match template TILE
    Tensor dst(newshape,res_type,Device::VULKAN);
    const Tensor& A = a.dtype() == res_type ? a : ops::Typecast(a,res_type);
    const Tensor& B = b.dtype() == res_type ? b : ops::Typecast(b,res_type);
    auto A_impl =  std::dynamic_pointer_cast<VKTensor>(A.get_impl());
    auto B_impl =  std::dynamic_pointer_cast<VKTensor>(B.get_impl());
    auto Dst_impl =  std::dynamic_pointer_cast<VKTensor>(dst.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<VulkanContext>(A_impl->context());
    MatmulParams params{
        .batch = uint32_t(batch),
        .N = uint32_t(cols),
        .K = uint32_t(common),
        .M = uint32_t(rows)
    };
    uint32_t gx = (cols + TILE_SZ - 1) / TILE_SZ;
    uint32_t gy = (rows + TILE_SZ - 1) / TILE_SZ;
    uint32_t gz = batch;
    ctx_impl->submitCompute(
        OpType::Matmul,
        res_type,
        {A_impl->buffer(),B_impl->buffer(),Dst_impl->buffer()},
        gx,gy,gz,
        &params,
        sizeof(params)
    );
    return dst;
}

template struct MulImpl<Device::VULKAN>;
}