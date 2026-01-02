#include <limits>
#include "backend/sycl/ops/concat.h"
using namespace via;

namespace ops {

template <typename T>
void concat_sycl(Tensor& output, const std::vector<Tensor>& tensors, int dim, sycl::queue& q) {
    T* out_data = static_cast<T*>(output.data());
    // 计算拼接偏移（元素个数）
    std::vector<size_t> offsets(tensors.size(), 0);
    for (size_t i = 1; i < tensors.size(); ++i) {
        offsets[i] = offsets[i - 1] + tensors[i - 1].shape()[dim];
    }
    // 每个偏移单位对应多少元素（例如 dim=1 时，dim 后面的元素乘积）
    size_t inner_dim = 1;
    for (size_t i = dim + 1; i < output.shape().size(); ++i) {
        inner_dim *= output.shape(i);
    }
    size_t stride = inner_dim;
    // 依次拷贝每个张量的数据
    for (size_t i = 0; i < tensors.size(); ++i) {
        const T* src_data = static_cast<const T*>(tensors[i].data());
        size_t copy_offset = offsets[i] * stride;
        size_t num_elements = tensors[i].numel();
        q.memcpy(out_data + copy_offset, src_data, num_elements * calc_dtype_size(output.dtype()));
    }
    q.wait();  // 同步
    
}

Tensor ConcatImpl<Device::SYCL>::execute(const std::vector<Tensor> &tensors, int dim) {
        const auto& first_shape = tensors[0].shape();
        DataType dtype = tensors[0].dtype();
        Device device = tensors[0].device();
        // 2. 计算输出张量的形状
        std::vector<int64_t> output_shape = first_shape;
        output_shape[dim] = 0;
        for (const auto& t : tensors) {
            output_shape[dim] += t.shape()[dim];
        }
        // 3. 创建输出张量
        // 4. 获取SYCL队列
        auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(tensors[0].get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();

        Tensor output(output_shape, dtype, device);
        dispatch_dtype(dtype, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            concat_sycl<T>(output, tensors, dim, q);
        });
        return output;
    }

    template struct ConcatImpl<Device::SYCL>;
}