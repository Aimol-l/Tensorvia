#include "backend/cuda/ops/transpose.h"
using namespace via;


namespace ops {

template <typename T>
__global__ void transpose_cuda2d(const T* RESTRICT src, T* RESTRICT dst, int rows, int cols) {
    constexpr int TILE_DIM = 32;
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 bank conflict
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    // 从全局内存加载到 shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = T(0);
    }
    __syncthreads();
    // 转置后写回全局内存
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x < rows && y < cols) {
        dst[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void TransposeImpl<Device::CUDA>::execute(Tensor& a){ 
    // 这个只支持二维转置
    const int rows = a.shape(0);
    const int cols = a.shape(1);
    Tensor output({cols, rows}, a.dtype(), a.device());
    // 分块尺寸
    constexpr int TILE_DIM = 32;
    dim3 threads(TILE_DIM, TILE_DIM);   // [32,32]
    dim3 blocks((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM); // [63,79]
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    dispatch_dtype(a.dtype(),[&](auto type_id){
        using T = typename decltype(type_id)::type;
        if constexpr(std::is_same_v<T,float16>){
            transpose_cuda2d<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(output.data()),rows,cols);
        }else if constexpr(std::is_same_v<T,bfloat16>){
            transpose_cuda2d<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(output.data()),rows,cols);
        }else{
            transpose_cuda2d<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const T*>(a.data()), static_cast<T*>(output.data()),rows,cols);
        }
    });
    ctx_impl->wait();
    a = std::move(output);
}

template <typename T, const int MAX_DIM = 8>
__global__ void transpose_cuda_nd(
    const T* in,
    T* out,
    const int* in_strides,
    const int* out_strides,
    const int* axes,
    int ndim,
    size_t numel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;

    // 计算多维坐标 old_idx
    int coord[MAX_DIM]; // 假设 <= 8 维
    int tmp = tid;
    for (int i = 0; i < ndim; ++i) {
        coord[i] = tmp / in_strides[i];
        tmp %= in_strides[i];
    }

    // 生成新坐标 new_coord
    int new_coord[MAX_DIM];
    for (int i = 0; i < ndim; ++i) {
        new_coord[i] = coord[axes[i]];
    }

    // 计算输出位置
    int out_index = 0;
    for (int i = 0; i < ndim; ++i) {
        out_index += new_coord[i] * out_strides[i];
    }

    out[out_index] = in[tid];
}

Tensor TransposeImpl<Device::CUDA>::execute(const Tensor& a, std::initializer_list<int64_t> axes) { 
    std::vector<int64_t> new_shape;
    for (auto axe : axes) new_shape.push_back(a.shape(axe));

    Tensor result(new_shape, a.dtype(), Device::CUDA);
    const int ndim = a.shape().size();
    const int numel = a.numel();
    // 计算输入和输出的步长
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * a.shape(i + 1);
        out_strides[i] = out_strides[i + 1] * result.shape(i + 1);
    }
    int* d_in_strides, * d_out_strides, * d_axes;//, *input_shape;
    cudaMallocManaged(&d_in_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_out_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_axes, sizeof(int) * ndim);
    // cudaMallocManaged(&input_shape, sizeof(int) * ndim);
    memcpy(d_in_strides, in_strides.data(), sizeof(int) * ndim);
    memcpy(d_out_strides, out_strides.data(), sizeof(int) * ndim);
    memcpy(d_axes, axes.begin(), sizeof(int) * ndim);
    // memcpy(input_shape, a.shape().data(), sizeof(int) * ndim);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    auto A = data_as_const_variant(a.dtype(), a.data());
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    std::visit([&](auto a_ptr) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(a_ptr)>>;
        transpose_cuda_nd<AType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const AType*>(a.data()), static_cast<AType*>(result.data()), d_in_strides, d_out_strides, d_axes, ndim, numel);
    },A);
    ctx_impl->wait();
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_axes);
    // cudaFree(input_shape);
    return result;
    
}
void TransposeImpl<Device::CUDA>::execute(const Tensor& a, Tensor& dst,std::initializer_list<int64_t> axes) {

    std::vector<int64_t> new_shape;
    for (auto axe : axes) new_shape.push_back(a.shape(axe));

    dst.reshape(new_shape);

    const int ndim = a.shape().size();
    const int numel = a.numel();
    // 计算输入和输出的步长
    std::vector<int64_t> in_strides(ndim, 1);
    std::vector<int64_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * a.shape(i + 1);
        out_strides[i] = out_strides[i + 1] * dst.shape(i + 1);
    }
    int* d_in_strides, * d_out_strides, * d_axes;//, *input_shape;
    cudaMallocManaged(&d_in_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_out_strides, sizeof(int) * ndim);
    cudaMallocManaged(&d_axes, sizeof(int) * ndim);
    // cudaMallocManaged(&input_shape, sizeof(int) * ndim);
    memcpy(d_in_strides, in_strides.data(), sizeof(int) * ndim);
    memcpy(d_out_strides, out_strides.data(), sizeof(int) * ndim);
    memcpy(d_axes, axes.begin(), sizeof(int) * ndim);
    // memcpy(input_shape, a.shape().data(), sizeof(int) * ndim);
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(a.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    dispatch_dtype(a.dtype(),[&](auto type_id){
        using T = typename decltype(type_id)::type;
        if constexpr(std::is_same_v<T,float16>){
            transpose_cuda_nd<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __half*>(a.data()), static_cast<__half*>(dst.data()), d_in_strides, d_out_strides, d_axes, ndim, numel);
        }else if constexpr(std::is_same_v<T,bfloat16>){
            transpose_cuda_nd<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const __nv_bfloat16*>(a.data()), static_cast<__nv_bfloat16*>(dst.data()), d_in_strides, d_out_strides, d_axes, ndim, numel);
        }else{
            transpose_cuda_nd<<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<const T*>(a.data()), static_cast<T*>(dst.data()), d_in_strides, d_out_strides, d_axes, ndim, numel);
        }
    });

    ctx_impl->wait();
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_axes);
    // cudaFree(input_shape);
}

template struct TransposeImpl<Device::CUDA>;

}