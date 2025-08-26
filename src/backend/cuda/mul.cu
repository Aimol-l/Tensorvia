#include "backend/cuda/ops/mul.h"

namespace ops {

// [w,k] @ [k,h] --> [w,h]
// [b,w,k] @ [b,k,h] --> [b,w,h]
template<typename T, const int TILE = 16>
__global__ void mul_cuda_3d(
    const T* a, const T* b, T* res,
    size_t batch, size_t w, size_t h, size_t k
) {
    // 每个 block 负责 TILE x TILE 的输出区域
    __shared__ T tile_a[TILE][TILE];
    __shared__ T tile_b[TILE][TILE];
    // 全局线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;  // 输出列块
    int by = blockIdx.y;  // 输出行块
    int bz = blockIdx.z;  // batch 索引
    // 当前 block 负责的输出起始位置
    int row = by * TILE + ty;
    int col = bx * TILE + tx;
    // 累加器（寄存器）
    T sum = T(0);
    // 遍历 k 维度，分块加载
    for (int tile_idx = 0; tile_idx < (k + TILE - 1) / TILE; ++tile_idx) {
        // 加载 tile_a: a[b][row][tile_k]
        int a_k = tile_idx * TILE + tx;
        if (row < w && a_k < k) {
            tile_a[ty][tx] = a[bz * w * k + row * k + a_k];
        } else {
            tile_a[ty][tx] = T(0);
        }
        // 加载 tile_b: b[b][tile_k][col]
        int b_k = tile_idx * TILE + ty;
        if (b_k < k && col < h) {
            tile_b[ty][tx] = b[bz * k * h + b_k * h + col];
        } else {
            tile_b[ty][tx] = T(0);
        }
        __syncthreads();
        // 内层计算：当前 tile 的乘积累加
        for (int i = 0; i < TILE; ++i) {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }
        __syncthreads();
    }
    // 写回结果
    if (row < w && col < h) {
        res[bz * w * h + row * w + col] = sum;
    }
}

Tensor MulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    // 外界的前置条件已经判定了shape的合法性
    auto a_shape = a.shape();
    size_t a_dim = a_shape.size();
    auto b_shape = b.shape();
    size_t b_dim = b_shape.size();
    int batch, w, h, k;
    std::vector<int> res_shape;
    if (a_dim == 2) {
        batch = size_t{1};
        w = a_shape[0];
        k = a_shape[1];
        h = b_shape[1];
        res_shape = {w, h};
    } else if (a_dim == 3) {
        batch = a_shape[0];
        w = a_shape[1];
        k = a_shape[2];
        h = b_shape[2];
        res_shape = {batch, w, h};
    } else {
        throw std::runtime_error("mul: invalid shape");
    }

    DataType res_type;
    if(a.dtype() > b.dtype()) {
        res_type = a.dtype();
    } else {
        res_type = b.dtype();
    }
    
    Tensor res(res_shape, res_type, Device::CUDA);
    dim3 threads(16, 16);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y, batch);
    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(res.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        if constexpr(std::is_same_v<AType, float16>){
            mul_cuda_3d<__half><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<const __half*>(b.data()), static_cast<__half*>(res.data()), batch, w, h, k);
        }else if constexpr(std::is_same_v<AType, bfloat16>) {
            mul_cuda_3d<__nv_bfloat16><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<const __nv_bfloat16*>(b.data()), static_cast<__nv_bfloat16*>(res.data()), batch, w, h, k);
        }else{
            mul_cuda_3d<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<const AType*>(b.data()), static_cast<AType*>(res.data()), batch, w, h, k);
        }
    },A);
    ctx_impl->wait();
    return res;
}

template struct MulImpl<Device::CUDA>;

}