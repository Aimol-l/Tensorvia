#include "backend/cpu/ops/slice.h"

#include <memory>



namespace ops {

template <typename T>
void slice_kernel(Tensor& out, const Tensor& in, const std::vector<std::pair<int, int>>& ranges) {
    auto t_shape = in.shape();
    size_t t_dim = t_shape.size();
    auto new_shape = out.shape();
    const size_t slice_dim = ranges.size();
    // 构建原始张量的 strides（行主序）
    std::vector<size_t> strides(t_dim, 1);
    for (int i = static_cast<int>(t_dim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * t_shape[i + 1];
    }
    // 获取数据指针
    const T* src_data = static_cast<const T*>(in.data());
    T* dest_data = static_cast<T*>(out.data());
    // 准备一个临时 index 缓冲区
    std::vector<int> indices(new_shape.size(), 0);
    const size_t total_elem = out.numel();
    for (size_t linear_idx = 0; linear_idx < total_elem; ++linear_idx) {
        // 将线性 idx 展开成多维坐标
        size_t remaining = linear_idx;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
            indices[i] = remaining % new_shape[i];
            remaining /= new_shape[i];
        }
        // 计算在原始张量中的偏移
        size_t src_offset = 0;
        for (size_t i = 0; i < t_dim; ++i) {
            int coord = (i < slice_dim) ? (ranges[i].first + indices[i]) : indices[i];
            src_offset += coord * strides[i];
        }
        dest_data[linear_idx] = src_data[src_offset];
    }
}

Tensor SliceImpl<Device::CPU>::execute(const Tensor& t, const std::vector<std::pair<int, int>>& ranges) {
    const auto& t_shape = t.shape();
    const size_t t_dim = t_shape.size();
    const size_t slice_dim = ranges.size();
    // 构建新 shape（切片维度为 [start, end)，非切片维度原样保留）
    std::vector<int> new_shape;
    new_shape.reserve(t_dim);
    for (size_t i = 0; i < slice_dim; ++i) {
        new_shape.push_back(ranges[i].second - ranges[i].first);
    }
    for (size_t i = slice_dim; i < t_dim; ++i) {
        new_shape.push_back(t_shape[i]);
    }
    // 创建输出 Tensor
    Tensor res(new_shape, t.dtype(), t.device());
    switch (t.dtype()) {
        case DataType::INT8:
            slice_kernel<int8_t>(res, t, ranges);
            break;
        case DataType::INT16:
            slice_kernel<int16_t>(res, t, ranges);
            break;
        case DataType::INT32:
            slice_kernel<int32_t>(res, t, ranges);
            break;
        case DataType::INT64:
            slice_kernel<int64_t>(res, t, ranges);
            break;
        case DataType::FLOAT16:
            slice_kernel<float16>(res, t, ranges);
            break;
        case DataType::FLOAT32:
            slice_kernel<float32>(res, t, ranges);
            break;
        case DataType::FLOAT64:
            slice_kernel<float64>(res, t, ranges);
            break;
        case DataType::BFLOAT16:
            slice_kernel<bfloat16>(res, t, ranges);
            break;
        default:
            throw std::runtime_error("slice: unsupported data type");
    }
    return res;
}

// 性能优化版本(未测试)
// static Tensor execute(const Tensor& t, const std::vector<std::pair<int, int>>& ranges) {
//     const auto& t_shape = t.shape();
//     const size_t t_dim = t_shape.size();
//     const size_t slice_dim = ranges.size();
//     // 1. 构建输出 shape
//     std::vector<int> new_shape;
//     new_shape.reserve(t_dim);
//     for (size_t i = 0; i < slice_dim; ++i)
//         new_shape.push_back(ranges[i].second - ranges[i].first);
//     for (size_t i = slice_dim; i < t_dim; ++i)
//         new_shape.push_back(t_shape[i]);
//     Tensor res(new_shape, t.dtype(), t.device());
//     // 2. 构建原张量 strides（行主序）
//     std::vector<size_t> strides(t_dim, 1);
//     for (int i = static_cast<int>(t_dim) - 2; i >= 0; --i)
//         strides[i] = strides[i + 1] * t_shape[i + 1];
//     // 3. 预先计算右开起始值
//     std::vector<int> starts;
//     starts.reserve(slice_dim);
//     for (const auto& [start, _] : ranges)
//         starts.push_back(start);
//     const size_t elem_size = calc_dtype_size(t.dtype());
//     const char* __restrict src_data = static_cast<const char*>(t.data());
//     char* __restrict dst_data = static_cast<char*>(res.data());
//     const auto& out_shape = res.shape();
//     // 4. 预计算每一维的 out_strides（帮助将线性idx转为多维索引）
//     std::vector<size_t> out_strides(out_shape.size(), 1);
//     for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i)
//         out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
//     // 5. 避免动态分配 indices，使用 stack 分配（性能好）
//     const size_t total = res.numel();
//     std::vector<int> coord(out_shape.size());
//     for (size_t linear_idx = 0; linear_idx < total; ++linear_idx) {
//         // 解码线性索引为多维坐标
//         size_t tmp = linear_idx;
//         for (size_t i = 0; i < out_shape.size(); ++i) {
//             coord[i] = tmp / out_strides[i];
//             tmp %= out_strides[i];
//         }
//         // 计算源 tensor 偏移（考虑 range 起点）
//         size_t src_offset = 0;
//         for (size_t i = 0; i < t_dim; ++i) {
//             int idx = (i < slice_dim) ? (starts[i] + coord[i]) : coord[i];
//             src_offset += idx * strides[i];
//         }
//         // 快速按元素复制
//         std::memcpy(dst_data + linear_idx * elem_size, src_data + src_offset * elem_size, elem_size);
//     }
//     return res;
// }

template struct SliceImpl<Device::CPU>;
}  // namespace ops