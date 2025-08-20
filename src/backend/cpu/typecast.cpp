#include "backend/cpu/ops/typecast.h"

#include "core/type_traits.h"
#include "cpu_tensor.h"


namespace ops {

template <typename TSrc, typename TDst>
void cast_kernel(const TSrc* src, TDst* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = static_cast<TDst>(src[i]);
    }
}

Tensor TypecastImpl<Device::CPU>::execute(const Tensor& a, DataType dst_type) {
    size_t size = a.numel();
    auto src_ptr = data_as_const_variant(a.dtype(), a.data());
    Tensor result(a.shape(), dst_type, Device::CPU);
    std::visit([&](auto* src_typed_ptr) {
        using SrcType = std::remove_pointer_t<decltype(src_typed_ptr)>;
        auto dispatch_dst = [&](auto* dst_typed_ptr) {
            using DstType = std::remove_pointer_t<decltype(dst_typed_ptr)>;
            cast_kernel<SrcType, DstType>(
                src_typed_ptr,
                dst_typed_ptr,
                size);
        };
        switch (dst_type) {
            case DataType::INT8:
                dispatch_dst(static_cast<int8_t*>(result.data()));
                break;
            case DataType::INT16:
                dispatch_dst(static_cast<int16_t*>(result.data()));
                break;
            case DataType::INT32:
                dispatch_dst(static_cast<int32_t*>(result.data()));
                break;
            case DataType::INT64:
                dispatch_dst(static_cast<int64_t*>(result.data()));
                break;
            case DataType::FLOAT16:
                dispatch_dst(static_cast<float16*>(result.data()));
                break;
            case DataType::BFLOAT16:
                dispatch_dst(static_cast<bfloat16*>(result.data()));
                break;
            case DataType::FLOAT32:
                dispatch_dst(static_cast<float32*>(result.data()));
                break;
            case DataType::FLOAT64:
                dispatch_dst(static_cast<float64*>(result.data()));
                break;
            default:
                throw std::runtime_error("Unsupported destination type");
        }
    },
               src_ptr);
    return result;
}

template struct TypecastImpl<Device::CPU>;
}  // namespace ops