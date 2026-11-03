#include <cstddef>
#include <span>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>
#include "_platform.h"

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"




namespace ops {

template <typename T>
void transpose_2d_kernel(const T* RESTRICT input, T* RESTRICT output, int64_t rows, int64_t cols) {
    constexpr int TILE_M = 64;  // 每块输出行数（可调）
    constexpr int TILE_N = 64; // 每块输出列数（尽量大）
    // OpenMP 并行：按输出行分块
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t j0 = 0; j0 < cols; j0 += TILE_M) {
        const int64_t j1 = std::min<int64_t>(j0 + TILE_M, cols);
        for (int64_t i0 = 0; i0 < rows; i0 += TILE_N) {
            const int64_t i1 = std::min<int64_t>(i0 + TILE_N, rows);
            // 转置当前块: output[j][i] = input[i][j]
            for (int64_t j = j0; j < j1; ++j) {
                T* out_row = output + j * rows + i0;      // 连续写入（缓存友好）
                const T* in_col = input + i0 * cols + j;  // 跨步读取
                for (int64_t i = i0; i < i1; ++i) {
                    out_row[i - i0] = in_col[(i - i0) * cols];
                }
            }
        }
    }
}
void TransposeImpl<Device::CPU>::execute(Tensor& a) {
    const int64_t rows = a.shape(0);
    const int64_t cols = a.shape(1);
    Tensor output({cols, rows}, a.dtype(), a.device());
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        transpose_2d_kernel<T>(
            static_cast<const T*>(a.data()),
            static_cast<T*>(output.data()),
            rows, cols
        );
    });
    a = std::move(output);
}

template <typename T>
void transpose_nd_kernel(T* dst_data,const T* src_data, const std::vector<int64_t>& axes, std::vector<int64_t> in_strides, std::vector<int64_t> out_strides, size_t size, int dim) {
    // 并行处理每个元素(使用OpenMP模拟并行)
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        int64_t tmp = static_cast<int64_t>(i);
        int64_t coord[kMaxTensorDims] = {};        // 解码后的多维坐标 (i,j,k, ...)
        // 解码坐标，将线性索引转换为多维坐标
        for (int d = 0; d < dim; ++d) {
            coord[d] = tmp / in_strides[d];  // 计算当前维度上的坐标
            tmp %= in_strides[d];            // 更新tmp，用于计算下一个维度上的坐标
        }
        // 按 axes 重排坐标并编码输出：out[d] 取输入的 axes[d] 维。
        int64_t out_index = 0;
        for (int d = 0; d < dim; ++d) {
            out_index += coord[axes[d]] * out_strides[d];
        }
        dst_data[out_index] = src_data[i];
    }
}
template <typename T>
void transpose_nd_kernel(T* dst, const T* src, std::span<const int64_t> shape, const std::vector<int64_t>& axes, int dim) {
    // 找到最外层不变的维度（axes[0] == 0, axes[1] == 1, ...）
    int prefix = 0;
    while (prefix < dim && axes[prefix] == prefix) {
        prefix++;
    }
    if (prefix == dim) {
        // 无需转置，直接拷贝
        std::memcpy(dst, src, sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<>()));
        return;
    }
    if (prefix == dim - 2) {
        // 最后两维需要转置 → 2D 分块转置
        const int64_t rows = shape[dim - 2];
        const int64_t cols = shape[dim - 1];
        const int64_t batch = std::accumulate(shape.begin(), shape.begin() + prefix, int64_t{1}, std::multiplies<>());
        #pragma omp parallel for
        for (int64_t b = 0; b < batch; ++b) {
            const T* src_batch = src + b * rows * cols;
            T* dst_batch = dst + b * cols * rows;
            transpose_2d_kernel(src_batch, dst_batch, rows, cols);
        }
        return;
    }
    // 其余情形（axes 既非恒等、也不是"前缀不变 + 末两维交换"的置换）：逐元素
    // 按 axes 重排。此前的递归按 "ax > 0" 过滤并排序内层 axes，丢掉了轴的先后
    // 关系，三维的 {1,0,2}、{2,0,1} 等会退化成恒等拷贝。
    std::vector<int64_t> in_strides(dim, 1);
    std::vector<int64_t> out_strides(dim, 1);
    for (int d = dim - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * shape[d + 1];
        out_strides[d] = out_strides[d + 1] * shape[axes[d + 1]];
    }
    const size_t total = static_cast<size_t>(
        std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>()));
    transpose_nd_kernel<T>(dst, src, axes, in_strides, out_strides, total, dim);
}
Tensor TransposeImpl<Device::CPU>::execute(const Tensor& a, std::initializer_list<int64_t> axes) {
    // 创建结果张量
    std::vector<int64_t> new_shape;
    for (auto axe : axes) new_shape.push_back(a.shape(axe));

    Tensor result(new_shape, a.dtype(), Device::CPU);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(result.data());
        const T* a_ptr = static_cast<const T*>(a.data());
        transpose_nd_kernel<T>(res_ptr,a_ptr,a.shape(),axes,a.shape().size());
    });
    return result;
}
void TransposeImpl<Device::CPU>::execute(const Tensor& a, Tensor& dst,std::initializer_list<int64_t> axes) {
    std::vector<int64_t> new_shape;
    for (auto axe : axes) new_shape.push_back(a.shape(axe));
    dst = dst.reshape(new_shape);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(dst.data());
        const T* a_ptr = static_cast<const T*>(a.data());
        transpose_nd_kernel<T>(res_ptr,a_ptr,a.shape(),axes,a.shape().size());
    });
}


// 速度慢
// Tensor TransposeImpl<Device::CPU>::execute(Tensor& a, std::initializer_list<int64_t> axes) {
//     // 创建结果张量
//     std::vector<int64_t> new_shape;
//     std::vector<int64_t> axes_v(axes);
//     for (auto axe : axes) new_shape.push_back(a.shape(axe));
//     Tensor result(new_shape, a.dtype(), Device::CPU);
//     const int dim = a.shape().size();
//     // 计算输入和输出的步长
//     std::vector<int64_t> in_strides(dim);
//     std::vector<int64_t> out_strides(dim);
//     in_strides[dim - 1] = 1;
//     out_strides[dim - 1] = 1;
//     for (int i = dim - 2; i >= 0; --i) {
//         in_strides[i] = in_strides[i + 1] * a.shape(i + 1);
//         out_strides[i] = out_strides[i + 1] * result.shape(i + 1);
//     }
//     dispatch_dtype(a.dtype(), [&](auto type_id) {
//         using T = typename decltype(type_id)::type;
//         transpose_nd_kernel<T>(static_cast<T*>(result.data()), static_cast<const T*>(a.data()),axes_v, in_strides, out_strides, a.numel(), dim);
//     });
//     return result;
// }




}  // namespace ops
