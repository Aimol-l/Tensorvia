#include "backend/cpu/ops/transpose.h"



namespace ops {

template <typename T>
void transpose_kernel(T* src_data, T* copy_data, int rows, int cols, int size) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            src_data[j * rows + i] = copy_data[i * cols + j];
        }
    }
}

template <typename T>
void transpose_kernel(T* dst_data, T* src_data, const std::vector<int64_t>& axes, std::vector<int64_t> in_strides, std::vector<int64_t> out_strides, size_t size, int dim) {
// 并行处理每个元素(使用OpenMP模拟并行)
#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int tmp = i;
        std::vector<int64_t> coord(dim);        // 存储解码后的多维坐标 (i,j,k, ...)
        std::vector<int64_t> trans_coord(dim);  // 存储重新排列后的多维坐标 (a,b,c, ...)
        // 解码坐标，将线性索引转换为多维坐标
        for (int d = 0; d < dim; ++d) {
            coord[d] = tmp / in_strides[d];  // 计算当前维度上的坐标
            tmp %= in_strides[d];            // 更新tmp，用于计算下一个维度上的坐标
        }
        // 根据axes重新排列坐标
        for (int d = 0; d < dim; ++d) {
            trans_coord[d] = coord[axes[d]];
        }
        // 编码坐标
        int out_index = 0;
        for (int d = 0; d < dim; ++d) {
            out_index += trans_coord[d] * out_strides[d];  // 根据新的坐标计算线性索引
        }
        dst_data[out_index] = src_data[i];
    }
}

void TransposeImpl<Device::CPU>::execute(Tensor& a) {
    switch (a.dtype()) {
        case DataType::FLOAT64:
            transpose_kernel<float64>(static_cast<float64*>(a.data()), static_cast<float64*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::FLOAT32:
            transpose_kernel<float32>(static_cast<float32*>(a.data()), static_cast<float32*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::FLOAT16:
            transpose_kernel<float16>(static_cast<float16*>(a.data()), static_cast<float16*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::BFLOAT16:
            transpose_kernel<bfloat16>(static_cast<bfloat16*>(a.data()), static_cast<bfloat16*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::INT64:
            transpose_kernel<int64_t>(static_cast<int64_t*>(a.data()), static_cast<int64_t*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::INT32:
            transpose_kernel<int32_t>(static_cast<int32_t*>(a.data()), static_cast<int32_t*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::INT16:
            transpose_kernel<int16_t>(static_cast<int16_t*>(a.data()), static_cast<int16_t*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        case DataType::INT8:
            transpose_kernel<int8_t>(static_cast<int8_t*>(a.data()), static_cast<int8_t*>(a.data()), a.shape(0), a.shape(1), a.numel());
            break;
        default:
            throw std::runtime_error("transpose not support this data type");
    }
    std::vector<int64_t> shape = {a.shape(1), a.shape(0)};
    a.reshape(shape);
}
Tensor TransposeImpl<Device::CPU>::execute(Tensor& a, std::initializer_list<int64_t> axes) {
    // 创建结果张量
    std::vector<int64_t> new_shape;
    std::vector<int64_t> axes_v(axes);
    for (auto axe : axes) new_shape.push_back(a.shape(axe));
    Tensor result(new_shape, a.dtype(), Device::CPU);
    const int dim = a.shape().size();
    // 计算输入和输出的步长
    std::vector<int64_t> in_strides(dim);
    std::vector<int64_t> out_strides(dim);
    in_strides[dim - 1] = 1;
    out_strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * a.shape(i + 1);
        out_strides[i] = out_strides[i + 1] * result.shape(i + 1);
    }
    switch (a.dtype()) {
        case DataType::FLOAT64:
            transpose_kernel<float64>(static_cast<float64*>(result.data()), static_cast<float64*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::FLOAT32:
            transpose_kernel<float32>(static_cast<float32*>(result.data()), static_cast<float32*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::FLOAT16:
            transpose_kernel<float16>(static_cast<float16*>(result.data()), static_cast<float16*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::BFLOAT16:
            transpose_kernel<bfloat16>(static_cast<bfloat16*>(result.data()), static_cast<bfloat16*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::INT64:
            transpose_kernel<int64_t>(static_cast<int64_t*>(result.data()), static_cast<int64_t*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::INT32:
            transpose_kernel<int32_t>(static_cast<int32_t*>(result.data()), static_cast<int32_t*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::INT16:
            transpose_kernel<int16_t>(static_cast<int16_t*>(result.data()), static_cast<int16_t*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        case DataType::INT8:
            transpose_kernel<int8_t>(static_cast<int8_t*>(result.data()), static_cast<int8_t*>(a.data()), axes_v, in_strides, out_strides, a.numel(), dim);
            break;
        default:
            throw std::runtime_error("transpose not support this data type");
    }
    return result;
}

template struct TransposeImpl<Device::CPU>;

}  // namespace ops