#include "factory.h"
#include "core/type_traits.h"
#include "backend/cpu/ops/typecast.h"

namespace ops {

// gcc/clang/icpx不需要，msvc有特殊实现的float16类
// std::float16_t float_to_float16(float f) {
//     // 处理特殊值（NaN, Inf, zero）
//     uint32_t f32 = std::bit_cast<uint32_t>(f);
//     uint16_t f16;
//     // IEEE 754 binary32 分解
//     uint32_t sign = (f32 >> 16) & 0x8000;
//     uint32_t exp  = (f32 >> 23) & 0xFF;
//     uint32_t mant = f32 & 0x7FFFFF;
//     if (exp == 0) {
//         // 零或 subnormal → 结果为 0（或 subnormal 半精度，但通常 flush to zero）
//         f16 = static_cast<uint16_t>(sign);
//     } else if (exp == 0xFF) {
//         // Inf 或 NaN
//         f16 = static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));
//     } else {
//         // 正常数
//         int32_t new_exp = exp - 127 + 15; // bias 调整: 127 → 15
//         if (new_exp <= 0) {
//             // underflow → flush to zero
//             f16 = static_cast<uint16_t>(sign);
//         } else if (new_exp >= 31) {
//             // overflow → Inf
//             f16 = static_cast<uint16_t>(sign | 0x7C00);
//         } else {
//             // 正常舍入（向偶数舍入，这里简化为截断 + 最近）
//             uint32_t rounded_mant = mant + 0x1000; // + 2^12 用于舍入
//             f16 = static_cast<uint16_t>(sign | (new_exp << 10) | (rounded_mant >> 13));
//         }
//     }
//     return std::bit_cast<std::float16_t>(f16);
// }
// void cast_kernel(const float32* src, float16* dst, size_t size) {
//     LOG_INFO("Typecast from float32 to float16");
//     for (size_t i = 0; i < size; i++) {
//         dst[i] = float_to_float16(src[i]);
//     }
// }


template <typename T, typename R>
void cast_kernel(const T* src, R* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = static_cast<R>(src[i]);
    }
}



void TypecastImpl<Device::CPU>::execute(Tensor& a, DataType dst_type) {
    auto src_ptr = data_as_const_variant(a.dtype(), a.data());

    auto cpu_impl = create_tensor_impl(a.numel(),a.dtype(), Device::CPU);
    std::visit([&](auto* src_typed_ptr) {
        using SrcType = std::remove_pointer_t<decltype(src_typed_ptr)>;
        auto dispatch_dst = [&](auto* dst_typed_ptr) {
            using DstType = std::remove_pointer_t<decltype(dst_typed_ptr)>;
            cast_kernel<SrcType, DstType>(
                src_typed_ptr,
                dst_typed_ptr,
                a.numel()
            );
        };
        dispatch_dtype(dst_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            dispatch_dst(static_cast<T*>(cpu_impl->data()));
        });
    },src_ptr);
    a.set_impl(cpu_impl,dst_type);
}

Tensor TypecastImpl<Device::CPU>::execute(const Tensor& a, DataType dst_type) {
    auto src_ptr = data_as_const_variant(a.dtype(), a.data());
    Tensor result(a.shape(), dst_type, Device::CPU);
    std::visit([&](auto* src_typed_ptr) {
        using SrcType = std::remove_pointer_t<decltype(src_typed_ptr)>;
        auto dispatch_dst = [&](auto* dst_typed_ptr) {
            using DstType = std::remove_pointer_t<decltype(dst_typed_ptr)>;
            cast_kernel<SrcType, DstType>(
                src_typed_ptr,
                dst_typed_ptr,
                a.numel()
            );
        };
        dispatch_dtype(dst_type, [&](auto type_id) {
            using T = typename decltype(type_id)::type;
            dispatch_dst(static_cast<T*>(result.data()));
        });
    },src_ptr);
    return result;
}



template struct TypecastImpl<Device::CPU>;
}  // namespace ops