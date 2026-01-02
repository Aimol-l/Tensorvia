#include "backend/cuda/ops/mul.h"
using namespace via;

namespace ops {

// [cols,k] @ [k,rows] --> [cols,rows]
// [b,cols,k] @ [b,k,rows] --> [b,cols,rows]
template <typename T, typename R, typename S, int TILE = 16>
__global__ void mul_cuda_3d(const T*  a_ptr, const R*  b_ptr, S* __restrict__ res_ptr,size_t batch, size_t cols, size_t rows, size_t k) {
    __shared__ T tile_a[TILE][TILE+1];
    __shared__ R tile_b[TILE][TILE+1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int row = by * TILE + ty; 
    int col = bx * TILE + tx;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    PromotedType sum = 0;
    int numTiles = (int)((k + TILE - 1) / TILE); //
    for (int tile_idx = 0; tile_idx < numTiles; ++tile_idx) {
        int a_k = tile_idx * TILE + tx; // 加载 A 的列索引 (k)
        if (row < rows && a_k < (int)k) {
            // A 被视作 [b, rows, k]
            tile_a[ty][tx] = a_ptr[bz * (size_t)rows * k + (size_t)row * k + a_k];
        } else {
            tile_a[ty][tx] = T(0);
        }
        int b_k = tile_idx * TILE + ty; // 加载 B 的行索引 (k)
        if (b_k < (int)k && col < cols) {
            // B 被视作 [b, k, cols]
            tile_b[ty][tx] = b_ptr[bz * (size_t)k * cols + (size_t)b_k * cols + col];
        } else {
            tile_b[ty][tx] = R(0);
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            // 显式 cast 到 accumulator 类型 S（防止 half 等精度问题）
            sum += static_cast<PromotedType>(tile_a[ty][i]) * static_cast<PromotedType>(tile_b[i][tx]);
        }
        __syncthreads();
    }
    if (row < rows && col < cols) {
        res_ptr[bz * (size_t)rows * cols + (size_t)row * cols + col] = static_cast<S>(sum);
    }
}

// 基本原则：输入的 shape 必须合法。
// eg. [3,2592,2048] @ [3,2048,2592] = [3,2592,2592]
Tensor MulImpl<Device::CUDA>::execute(const Tensor& a, const Tensor& b) {
    int batch =     a.shape().size() == 3?a.shape(0):1; // 3
    int rows =      a.shape().size() == 3?a.shape(1):a.shape(0); //2592
    int common =    a.shape().size() == 3?a.shape(2):a.shape(1); // 2048
    int cols =      a.shape().size() == 3?b.shape(2):b.shape(1); // 2592
    std::vector<int64_t> newshape;
    if(a.shape().size() == 3){
        newshape = {batch,rows,cols};
    }else{
        newshape = {rows,cols};
    }
    DataType res_type = compute_type(a.dtype(),b.dtype());
    Tensor result(newshape, res_type, Device::CUDA);

    constexpr int TILE_SZ = 16; // or match template TILE
    dim3 threads(TILE_SZ, TILE_SZ);
    // blocks = [162,162,3]
    dim3 blocks((cols + TILE_SZ - 1) / TILE_SZ,(rows + TILE_SZ - 1) / TILE_SZ,batch);

    auto src_impl =  std::dynamic_pointer_cast<CUDATensor>(result.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());

    auto c_visitor = [&]<typename T, typename R>(const T* a_ptr,const R* b_ptr) {
        switch (res_type) {
            case DataType::INT8:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<int8_t*>(result.data()), batch, cols, rows, common); break;
            case DataType::INT16:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<int16_t*>(result.data()), batch, cols, rows, common); break;
            case DataType::INT32:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<int32_t*>(result.data()), batch, cols, rows, common); break;
            case DataType::INT64:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<int64_t*>(result.data()), batch, cols, rows, common); break;
            case DataType::FLOAT16:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<__half*>(result.data()), batch, cols, rows, common); break;
            case DataType::BFLOAT16:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<__nv_bfloat16*>(result.data()), batch, cols, rows, common); break;
            case DataType::FLOAT32:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<float*>(result.data()), batch, cols, rows, common); break;
            case DataType::FLOAT64:
                mul_cuda_3d<<<blocks,threads,0,ctx_impl->stream()>>>(a_ptr,b_ptr,static_cast<double*>(result.data()), batch, rows, rows, common); break;
            default: throw std::runtime_error("Unsupported destination dtype");
        }
    };

    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());

    std::visit([&](auto a_ptr,auto b_ptr) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        using R = std::remove_cv_t<std::remove_pointer_t<decltype(b_ptr)>>;
        if constexpr(std::is_same_v<T,float16> && std::is_same_v<R,float16>){
            c_visitor(static_cast<const __half*>(a.data()),static_cast<const __half*>(b.data()));
        }else if constexpr(std::is_same_v<T,bfloat16> && std::is_same_v<R,bfloat16>){
            c_visitor(static_cast<const __nv_bfloat16*>(a.data()),static_cast<const __nv_bfloat16*>(b.data()));
        }else if constexpr(std::is_same_v<T,float16> && std::is_same_v<R,bfloat16>){
            c_visitor(static_cast<const __half*>(a.data()),static_cast<const __nv_bfloat16*>(b.data()));
        }else if constexpr(std::is_same_v<T,bfloat16> && std::is_same_v<R,float16>){
            c_visitor(static_cast<const __nv_bfloat16*>(a.data()),static_cast<const __half*>(b.data()));
        }else if constexpr(std::is_same_v<T,float16>){
            c_visitor(static_cast<const __half*>(a.data()),static_cast<const R*>(b.data()));
        }else if constexpr(std::is_same_v<R,float16>){
            c_visitor(static_cast<const T*>(a.data()),static_cast<const __half*>(b.data()));
        }else if constexpr(std::is_same_v<T,bfloat16>){
            c_visitor(static_cast<const __nv_bfloat16*>(a.data()),static_cast<const R*>(b.data()));
        }else if constexpr(std::is_same_v<R,bfloat16>){
            c_visitor(static_cast<const T*>(a.data()),static_cast<const __nv_bfloat16*>(b.data()));
        }else{
            c_visitor(static_cast<const T*>(a.data()),static_cast<const R*>(b.data()));
        }
    },A,B);
    ctx_impl->wait();
    return result;
}

template struct MulImpl<Device::CUDA>;

}