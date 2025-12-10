#include "backend/cuda/ops/concat.h"

namespace ops {

// CUDA核函数实现 - 单核函数处理所有张量
template <typename T>
__global__ void concat_cuda(
    T* output, 
    const T** inputs, 
    const int* offsets,
    const int* res_coord_weights,
    const int* all_strides, // 所有输入张量的步长连续存储
    const int* all_shapes,  // 所有输入张量的形状连续存储
    const size_t* numels,   // 每个输入张量的元素数量
    int dim, // 指定合并的维度
    size_t t_dim, //输入张量的维度
    int tensor_num
) {
    // 计算全局线程索引
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算总元素数
    size_t total_elements = 0;
    for (int i = 0; i < tensor_num; ++i) {
        total_elements += numels[i];
    }
    // 如果索引超出总元素数，直接返回
    if (global_idx >= total_elements) return;
    // 确定当前线程处理哪个张量的哪个元素
    size_t remaining_idx = global_idx;
    int tensor_idx = 0;
    // 找到当前元素属于哪个张量
    for (; tensor_idx < tensor_num; ++tensor_idx) {
        if (remaining_idx < numels[tensor_idx]) {
            break;
        }
        remaining_idx -= numels[tensor_idx];
    }
    // 获取当前张量的相关信息
    const int* strides = all_strides + tensor_idx * t_dim;
    const void* input_ptr = inputs[tensor_idx];
    // 计算当前元素在输入张量中的坐标
    size_t temp = remaining_idx;
    int coord[8]; // 假设最大维度为8，可根据实际情况调整
    for (int k = 0; k < t_dim; ++k) {
        coord[k] = temp / strides[k];
        temp %= strides[k];
    }
    // 调整拼接维度的坐标
    coord[dim] += offsets[tensor_idx];
    // 计算在输出张量中的线性坐标
    int linear_coord = 0;
    for (int k = 0; k < t_dim; ++k) {
        linear_coord += coord[k] * res_coord_weights[k];
    }
    // 赋值操作
    output[linear_coord] = static_cast<const T*>(input_ptr)[remaining_idx];
}

Tensor ConcatImpl<Device::CUDA>::execute(const std::vector<Tensor>& tensors, int dim) {
    // 确定输出张量的类型，最高精度
    auto res_type = tensors[0].dtype();
    // 计算输出张量的形状
    std::vector<int64_t> out_shape = tensors[0].shape();
    size_t concat_size = 0;
    for (auto& t : tensors) {
        concat_size += t.shape()[dim];
        // res_type = std::max(res_type, t.dtype());
    }
    size_t num_tensor = tensors.size();

    out_shape[dim] = concat_size;
    Tensor res(out_shape, res_type, Device::CUDA);

    // 获取维度信息
    size_t t_dim = tensors[0].shape().size();

    // 计算输出张量的坐标权重（步长）
    std::vector<int64_t> res_coord_weight(t_dim, 1);
    for (int i = t_dim - 2; i >= 0; --i) {
        res_coord_weight[i] = res_coord_weight[i + 1] * out_shape[i + 1];
    }

    // 计算每个输入张量的偏移量
    std::vector<int64_t> offsets(num_tensor, 0);
    for (int i = 1; i < num_tensor; ++i) {
        offsets[i] = offsets[i - 1] + tensors[i - 1].shape()[dim];
    }

    // 预计算每个输入张量的步长和元素数量
    std::vector<int64_t> all_strides;
    std::vector<int64_t> all_shapes;
    std::vector<size_t> all_numels;
    std::vector<const void*> input_ptrs;
    
    // 计算总元素数
    size_t total_elements = 0;
    for (int i = 0; i < num_tensor; ++i) {
        auto t_shape = tensors[i].shape();
        size_t numel = tensors[i].numel();
        all_numels.push_back(numel);
        total_elements += numel;
        
        // 存储数据指针（转换为void*）
        input_ptrs.push_back(tensors[i].data());
        
        // 计算当前张量的步长
        std::vector<int64_t> stride(t_dim, 1);
        for (int k = t_dim - 2; k >= 0; --k) {
            stride[k] = stride[k + 1] * t_shape[k + 1];
        }
        
        // 将步长和形状添加到连续数组中
        all_strides.insert(all_strides.end(), stride.begin(), stride.end());
        all_shapes.insert(all_shapes.end(), t_shape.begin(), t_shape.end());
    }

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    auto src_ptr = std::dynamic_pointer_cast<CUDATensor>(res.get_impl());
    auto ctx_impl = std::dynamic_pointer_cast<CUDAContext>(src_ptr->context());

    int *d_all_strides, *d_all_shapes, *d_res_coord_weights, *d_offsets;
    size_t *d_numels;
    const void** d_inputs;

    cudaMallocManaged(&d_all_strides, all_strides.size() * sizeof(int));
    cudaMallocManaged(&d_all_shapes, all_shapes.size() * sizeof(int));
    cudaMallocManaged(&d_numels, all_numels.size() * sizeof(size_t));
    cudaMallocManaged(&d_res_coord_weights, res_coord_weight.size() * sizeof(int));
    cudaMallocManaged(&d_offsets, offsets.size() * sizeof(int));
    cudaMallocManaged(&d_inputs, num_tensor * sizeof(void*));

    memcpy(d_all_strides, all_strides.data(), all_strides.size() * sizeof(int));
    memcpy(d_all_shapes, all_shapes.data(), all_shapes.size() * sizeof(int));
    memcpy(d_numels, all_numels.data(), all_numels.size() * sizeof(size_t));
    memcpy(d_res_coord_weights, res_coord_weight.data(), res_coord_weight.size() * sizeof(int));
    memcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int));
    memcpy(d_inputs, input_ptrs.data(), num_tensor * sizeof(const void*));

    // 调用核函数
    auto R = data_as_const_variant(res.dtype(), res.data());

    std::visit([&](auto r_ptr) {
        using RType = std::remove_cv_t<std::remove_pointer_t<decltype(r_ptr)>>;
        concat_cuda<RType><<<blocks, threads, 0, ctx_impl->stream()>>>(static_cast<RType*>(res.data()), reinterpret_cast<const RType**>(d_inputs), d_offsets, d_res_coord_weights, d_all_strides, d_all_shapes, d_numels, dim, t_dim, num_tensor);
    },R);

    ctx_impl->wait();
    cudaFree(d_all_strides);
    cudaFree(d_all_shapes);
    cudaFree(d_numels);
    cudaFree(d_res_coord_weights);
    cudaFree(d_offsets);
    cudaFree(d_inputs);

    return res;
}

template struct ConcatImpl<Device::CUDA>;
}  // namespace ops