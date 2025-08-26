#include "backend/sycl/ops/typecast.h"

namespace ops {

    template <typename TSrc, typename TDst>
    void cast_sycl(const TSrc* src, TDst* dst, size_t size, sycl::queue& q) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
                dst[i] = static_cast<TDst>(src[i]);
            });
        }).wait();
    }


Tensor TypecastImpl<Device::SYCL>::execute(const Tensor& a, DataType dst_type) {
        Tensor result(a.shape(), dst_type, Device::SYCL);
        auto src_ptr = data_as_const_variant(a.dtype(),a.data());
        size_t size = a.numel();
        auto src_impl =  std::dynamic_pointer_cast<SYCLTensor>(a.get_impl());
        auto ctx_impl = std::dynamic_pointer_cast<SYCLContext>(src_impl->context());
        auto& q = ctx_impl->get_queue(); 
        std::visit([&](auto* src_typed_ptr) {
            using SrcType = std::remove_pointer_t<decltype(src_typed_ptr)>;
            auto dispatch_dst = [&](auto* dst_typed_ptr) {
                using DstType = std::remove_pointer_t<decltype(dst_typed_ptr)>;
                cast_sycl<SrcType, DstType>(
                    src_typed_ptr,
                    dst_typed_ptr,
                    size,
                    q
                );
            };
            switch (dst_type) {
                case DataType::INT8:    dispatch_dst(static_cast<int8_t*>(result.data())); break;
                case DataType::INT16:   dispatch_dst(static_cast<int16_t*>(result.data())); break;
                case DataType::INT32:   dispatch_dst(static_cast<int32_t*>(result.data())); break;
                case DataType::INT64:   dispatch_dst(static_cast<int64_t*>(result.data())); break;
                case DataType::FLOAT16: dispatch_dst(static_cast<float16*>(result.data())); break;
                case DataType::BFLOAT16:dispatch_dst(static_cast<bfloat16*>(result.data())); break;
                case DataType::FLOAT32: dispatch_dst(static_cast<float32*>(result.data())); break;
                case DataType::FLOAT64: dispatch_dst(static_cast<float64*>(result.data())); break;
                default: throw std::runtime_error("Unsupported destination type");
            }
        }, src_ptr);
        return result;
    }
template struct TypecastImpl<Device::SYCL>;
}