#include "backend/cpu/ops/concat.h"



namespace ops {

template <typename T>
void concat_kernel(Tensor& res, const std::vector<Tensor>& tensors, int dim) {
    const auto& out_shape = res.shape();
    size_t t_dim = tensors[0].shape().size();
    size_t tensor_num = tensors.size();
    T* pr = static_cast<T*>(res.data());
    auto res_elem_size = calc_dtype_size(res.dtype());
    std::vector<size_t> res_coord_weight(t_dim, 1);
    for (int i = t_dim - 2; i >= 0; --i) {
        res_coord_weight[i] = res_coord_weight[i + 1] * out_shape[i + 1];  // 固定
    }
    std::vector<size_t> offsets(tensor_num, 0);
    for (int i = 1; i < tensor_num; ++i) {
        offsets[i] = offsets[i - 1] + tensors[i - 1].shape()[dim];
    }
    for (int i = 0; i < tensor_num; ++i) {
        auto t_shape = tensors[i].shape();
        const T* pt = static_cast<const T*>(tensors[i].data());
        size_t t_numl = tensors[i].numel();
        auto src_elem_size = calc_dtype_size(tensors[i].dtype());
        std::vector<int64_t> stride(t_dim, 1);
        for (int k = t_dim - 2; k >= 0; --k)
            stride[k] = stride[k + 1] * t_shape[k + 1];
        for (size_t j = 0; j < t_numl; ++j) {
            size_t temp = j;
            std::vector<size_t> coord(t_dim);
            for (int k = 0; k < t_dim; ++k) {
                coord[k] = temp / stride[k];
                temp %= stride[k];
            }
            coord[dim] += offsets[i];
            int linear_coord = 0;
            for (int k = 0; k < t_dim; ++k) {
                linear_coord += coord[k] * res_coord_weight[k];
            }
            pr[linear_coord] = static_cast<T>(pt[j]);
        }
    }
}

Tensor ConcatImpl<Device::CPU>::execute(const std::vector<Tensor>& tensors, int dim) {
    // 确定输出张量的类型，最高精度
    auto res_type = tensors[0].dtype();
    // 计算输出张量的形状
    std::vector<int64_t> out_shape = tensors[0].shape();
    size_t concat_size = 0;
    for (auto& t : tensors) {
        concat_size += t.shape(dim);
        res_type = std::max(res_type, t.dtype());
    }
    out_shape[dim] = concat_size;
    Tensor res(out_shape, res_type, Device::CPU);
    dispatch_dtype(res_type, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        concat_kernel<T>(res,tensors,dim);
    });
    return res;
}

template struct ConcatImpl<Device::CPU>;
}  // namespace ops