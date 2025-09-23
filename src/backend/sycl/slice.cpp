
#include "backend/sycl/ops/slice.h"

namespace ops {

    template <typename T, int MAX_DIMS = 8>
    void slice_sycl(Tensor& output, const Tensor& input, const std::vector<std::pair<int, int>>& ranges, sycl::queue& q) {
        const T* in_data = static_cast<const T*>(input.data());
        T* out_data = static_cast<T*>(output.data());

        const int ndim = static_cast<int>(input.shape().size());

        // 用 sycl::malloc_shared 分配 POD 数组
        int* in_shape = sycl::malloc_shared<int>(MAX_DIMS, q);
        int* out_shape = sycl::malloc_shared<int>(MAX_DIMS, q);
        int* in_strides = sycl::malloc_shared<int>(MAX_DIMS, q);
        int* start_idx = sycl::malloc_shared<int>(MAX_DIMS, q);

        for (int i = 0; i < ndim; ++i) {
            in_shape[i] = input.shape(i);
            out_shape[i] = output.shape(i);
            start_idx[i] = ranges[i].first;
        }
        in_strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
        }
        size_t numel = output.numel();
        q.submit([=](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
                size_t linear_idx = idx[0];
                int out_coords[MAX_DIMS];
                for (int i = ndim - 1; i >= 0; --i) {
                    out_coords[i] = linear_idx % out_shape[i];
                    linear_idx /= out_shape[i];
                }
                size_t in_linear_idx = 0;
                for (int i = 0; i < ndim; ++i) {
                    int in_coord = out_coords[i] + start_idx[i];
                    in_linear_idx += in_coord * in_strides[i];
                }
                out_data[idx[0]] = in_data[in_linear_idx];
            });
        }).wait();

        // 释放共享内存
        sycl::free(in_shape, q);
        sycl::free(out_shape, q);
        sycl::free(in_strides, q);
        sycl::free(start_idx, q);
    }


Tensor SliceImpl<Device::SYCL>::execute(const Tensor& t, const std::vector<std::pair<int, int>>& ranges){
        // 计算新的shape
        std::vector<int64_t> new_shape;
        for (size_t i = 0; i < ranges.size(); ++i) {
            const auto& [start, end] = ranges[i];
            new_shape.push_back(end - start);
        }
        // 创建新的Tensor
        Tensor res(new_shape, t.dtype(), t.device());
        // 4. 获取SYCL队列
        auto src_impl = std::dynamic_pointer_cast<SYCLTensor>(t.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        // 5. 调用对应的内核函数
        switch (t.dtype()) {
            case DataType::INT8:     slice_sycl<int8_t>(res,  t, ranges, q); break;
            case DataType::INT16:    slice_sycl<int16_t>(res, t, ranges, q); break;
            case DataType::INT32:    slice_sycl<int32_t>(res, t, ranges, q); break;
            case DataType::INT64:    slice_sycl<int64_t>(res, t, ranges, q); break;
            case DataType::FLOAT16:  slice_sycl<float16>(res, t, ranges, q); break;
            case DataType::FLOAT32:  slice_sycl<float32>(res, t, ranges, q); break;
            case DataType::FLOAT64:  slice_sycl<float64>(res, t, ranges, q); break;
            case DataType::BFLOAT16: slice_sycl<bfloat16>(res, t, ranges, q); break;
            default: throw std::runtime_error("slice: unsupported data type");
        }
        return res;
    }

template struct SliceImpl<Device::SYCL>;

}