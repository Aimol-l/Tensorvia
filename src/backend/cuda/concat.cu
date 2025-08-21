#include "backend/cuda/ops/concat.h"

namespace ops {

template <typename T>
__global__ void concat_cuda(
    T* res,
    const T** tensors,
    size_t* inputs_size,
    size_t res_size,
    int dim,
    int outer_size,
    int inner_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= res_size) return;
    int tensor_idx = 0;
    size_t offset = 0;
    while (tensor_idx < gridDim.y && idx >= offset + inputs_size[tensor_idx]) {
        offset += inputs_size[tensor_idx];
        ++tensor_idx;
    }
    if (tensor_idx >= gridDim.y) return;
    int local_idx = idx - offset;
    int inner = local_idx % inner_size;
    int outer = local_idx / inner_size;
    int input_index = outer * inputs_size[tensor_idx] / inner_size + inner;
    res[idx] = tensors[tensor_idx][input_index];
}

Tensor ConcatImpl<Device::CUDA>::execute(const std::vector<Tensor> &tensors, int dim){
    DataType dtype = tensors[0].dtype();
    Device device = tensors[0].device();
    // 2. 计算输出张量的形状
    std::vector<int> res_shape = tensors[0].shape();
    res_shape[dim] = 0;
    for (const auto& t : tensors) {
        res_shape[dim] += t.shape(dim);
    }
    // 3. 创建输出张量
    Tensor res(res_shape, dtype, device);
    // 4. 计算 outer_size 和 inner_size
    int outer_size = std::accumulate(res_shape.begin(), res_shape.begin() + dim, 1, std::multiplies<int>());
    int inner_size = std::accumulate(res_shape.begin() + dim + 1, res_shape.end(), 1, std::multiplies<int>());
    // 5. 计算线程配置
    constexpr size_t threads = 256;
    size_t blocks = (res.numel() + threads - 1) / threads;
    // 6. 提取输入张量指针和大小
    std::vector<const void*> host_tensors;
    std::vector<size_t> input_sizes;
    for (const auto& t : tensors) {
        host_tensors.push_back(t.data());
        input_sizes.push_back(t.shape(dim));
    }
    // 7. 在设备上分配指针数组
    const void** device_tensors = nullptr;
    cudaMalloc(&device_tensors, host_tensors.size() * sizeof(void*));
    cudaMemcpy((void**)device_tensors, host_tensors.data(), host_tensors.size() * sizeof(void*), cudaMemcpyHostToDevice);
    // 8. 分配并拷贝 input_sizes
    size_t* device_sizes = nullptr;
    cudaMalloc(&device_sizes, input_sizes.size() * sizeof(size_t));
    cudaMemcpy(device_sizes, input_sizes.data(), input_sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    auto src_impl = std::dynamic_pointer_cast<CUDATensor>(tensors[0].get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_impl->context());

    auto Res = data_as_const_variant(res.dtype(), res.data());

    std::visit([&](auto res_ptr){
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(res_ptr)>>; // const T* -> const T -> T
        if constexpr(std::is_same_v<AType,float16>){
            concat_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(
                static_cast<__half*>(res.data()),
                reinterpret_cast<const __half**>(device_tensors),
                device_sizes,
                res.numel(),
                dim,
                outer_size,
                inner_size
            );
        }else if constexpr(std::is_same_v<AType,bfloat16>){
            concat_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(
                static_cast<__nv_bfloat16*>(res.data()),
                reinterpret_cast<const __nv_bfloat16**>(device_tensors),
                device_sizes,
                res.numel(),
                dim,
                outer_size,
                inner_size
            );
        }else{
            concat_cuda<<<blocks, threads, 0, ctx_impl->stream()>>>(
                static_cast<AType*>(res.data()),
                reinterpret_cast<const AType**>(device_tensors),
                device_sizes,
                res.numel(),
                dim,
                outer_size,
                inner_size
            );
        }
    },Res);
    ctx_impl->wait();
    return res;
}

template struct ConcatImpl<Device::CUDA>;
}