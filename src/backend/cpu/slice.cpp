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
    std::vector<int64_t> indices(new_shape.size(), 0);
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
    std::vector<int64_t> new_shape;
    new_shape.reserve(t_dim);
    for (size_t i = 0; i < slice_dim; ++i) {
        new_shape.push_back(ranges[i].second - ranges[i].first);
    }
    for (size_t i = slice_dim; i < t_dim; ++i) {
        new_shape.push_back(t_shape[i]);
    }
    // 创建输出 Tensor
    Tensor res(new_shape, t.dtype(), t.device());
    dispatch_dtype(res.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        slice_kernel<T>(res, t, ranges);
    });
    return res;
}
template struct SliceImpl<Device::CPU>;
}  // namespace ops