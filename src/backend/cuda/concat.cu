#include "backend/cuda/ops/concat.h"

namespace ops {

// template <typename T>
// __global__ void concat_cuda(
//     T* res,
//     const T** tensors,
//     size_t* inputs_size,
//     size_t res_size,
//     int dim,
//     int outer_size,
//     int inner_size
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= res_size) return;
//     int tensor_idx = 0;
//     size_t offset = 0;
//     while (tensor_idx < gridDim.y && idx >= offset + inputs_size[tensor_idx]) {
//         offset += inputs_size[tensor_idx];
//         ++tensor_idx;
//     }
//     if (tensor_idx >= gridDim.y) return;
//     int local_idx = idx - offset;
//     int inner = local_idx % inner_size;
//     int outer = local_idx / inner_size;
//     int input_index = outer * inputs_size[tensor_idx] / inner_size + inner;
//     res[idx] = tensors[tensor_idx][input_index];
// }

Tensor ConcatImpl<Device::CUDA>::execute(const std::vector<Tensor> &tensors, int dim){
    return tensors[0].clone();
}

template struct ConcatImpl<Device::CUDA>;

}