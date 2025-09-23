
#include "backend/sycl/ops/transpose.h"

namespace ops {

    template <typename T>
    void transpose_sycl(Tensor& a,sycl::queue& q) {
        int rows = a.shape(0);
        int cols = a.shape(1);
        size_t size = a.numel();
        // 复制原始数据
        T* src_data = static_cast<T*>(a.data());
        T* copy_data = sycl::malloc_device<T>(size,q);
        q.memcpy(copy_data,src_data,size*sizeof(T)).wait();
        // 转置
        q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx){
                size_t i = idx[0];
                size_t j = idx[1];
                src_data[j * rows + i] = copy_data[i * cols + j];
            });
        }).wait();
        sycl::free(copy_data,q);
    }
    
    template <typename T>
    void transpose_sycl(Tensor& result,Tensor& a,std::vector<int64_t> axes,sycl::queue& q) {
        int dim = a.shape().size();
        T* src_data = static_cast<T*>(a.data());
        T* dst_data = static_cast<T*>(result.data());
        int* axes_v = sycl::malloc_shared<int>(dim,q);
        int* in_strides = sycl::malloc_shared<int>(dim,q);
        int* out_strides = sycl::malloc_shared<int>(dim,q);
        // 初始化 strides
        in_strides[dim - 1] = 1;
        out_strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i) {
            in_strides[i] = in_strides[i + 1] * a.shape(i + 1);
            out_strides[i] = out_strides[i + 1] * result.shape(i + 1);
        }
        for(int i =0;i<dim;i++) axes_v[i] = axes[i];
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(a.numel()), [=](sycl::id<1> idx){
                auto i = idx[0];
                auto tmp = i;
                int coord[4] = {0, 0, 0, 0};
                int trans_coord[4] = {0, 0, 0, 0};
                // 解码坐标
                for (int d = 0; d < dim; ++d) {
                    coord[d] = tmp / in_strides[d];
                    tmp %= in_strides[d];
                }
                // 转换坐标
                for (int d = 0; d < dim; ++d) {
                    trans_coord[d] = coord[axes_v[d]];
                }
                // 编码坐标
                int out_index = 0;
                for (int d = 0; d < dim; ++d) {
                    out_index += trans_coord[d] * out_strides[d];
                }
                dst_data[out_index] = src_data[i];
            });
        });
        q.wait(); // 等待计算完成
        sycl::free(axes_v,q);
        sycl::free(in_strides,q);
        sycl::free(out_strides,q);
    }
    



     void TransposeImpl<Device::SYCL>::execute(Tensor& a){
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        switch (a.dtype()) {
            case DataType::INT8:    transpose_sycl<int8_t>(a, q);break;
            case DataType::INT16:   transpose_sycl<int16_t>(a, q); break;
            case DataType::INT32:   transpose_sycl<int32_t>(a, q); break;
            case DataType::INT64:   transpose_sycl<int64_t>(a, q); break;
            case DataType::FLOAT16: transpose_sycl<float16>(a, q); break;
            case DataType::FLOAT32: transpose_sycl<float32>(a, q); break;
            case DataType::FLOAT64: transpose_sycl<float64>(a, q); break;
            case DataType::BFLOAT16:transpose_sycl<bfloat16>(a, q); break;
        }
        std::vector<int64_t> shape = {a.shape(1),a.shape(0)};
        a.reshape(shape);

    }
     Tensor TransposeImpl<Device::SYCL>::execute(Tensor& a,std::initializer_list<int64_t> axes){
        // 创建结果张量
        std::vector<int64_t> new_shape;
        std::vector<int64_t> axes_v(axes);
        for(auto axe:axes)  new_shape.push_back(a.shape(axe));
        Tensor result(new_shape,a.dtype(),Device::SYCL);
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue();
        switch (a.dtype()) {
            case DataType::INT8:    transpose_sycl<int8_t>(result,a,axes_v,q);break;
            case DataType::INT16:   transpose_sycl<int16_t>(result,a,axes_v,q); break;
            case DataType::INT32:   transpose_sycl<int32_t>(result,a,axes_v,q); break;
            case DataType::INT64:   transpose_sycl<int64_t>(result,a,axes_v,q); break;
            case DataType::FLOAT16: transpose_sycl<float16>(result,a,axes_v,q); break;
            case DataType::FLOAT32: transpose_sycl<float32>(result,a,axes_v,q); break;
            case DataType::FLOAT64: transpose_sycl<float64>(result,a,axes_v,q); break;
            case DataType::BFLOAT16:transpose_sycl<bfloat16>(result,a,axes_v,q); break;
        }
        return result;
    }
    template struct TransposeImpl<Device::SYCL>;
    }  // namespace ops