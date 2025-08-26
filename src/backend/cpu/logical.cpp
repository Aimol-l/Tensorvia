#include "backend/cpu/ops/logical.h"



namespace ops {

// 定义一个枚举类来区分不同的比较操作
enum class CmpOp {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
};
template <CmpOp Op, typename T, typename R>
void comparison_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
#pragma omp parallel for if (size > 4096)  // 4096 是一个经验值
    for (int i = 0; i < size; ++i) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            // 对整数直接进行比较，速度最快
            const auto val_a = a_ptr[i];
            const auto val_b = b_ptr[i];
            if constexpr (Op == CmpOp::Equal)
                res[i] = (val_a == val_b);
            else if constexpr (Op == CmpOp::NotEqual)
                res[i] = (val_a != val_b);
            else if constexpr (Op == CmpOp::Greater)
                res[i] = (val_a > val_b);
            else if constexpr (Op == CmpOp::GreaterEqual)
                res[i] = (val_a >= val_b);
            else if constexpr (Op == CmpOp::Less)
                res[i] = (val_a < val_b);
            else if constexpr (Op == CmpOp::LessEqual)
                res[i] = (val_a <= val_b);

        } else {
            using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            if constexpr (Op == CmpOp::Equal || Op == CmpOp::NotEqual) {
                // 定义容差，逻辑保持不变：由原始类型中精度最低的决定
                using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
                constexpr PromotedType abs_tol = [] {
                    if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-9);
                    else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-5);
                    else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
                    else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
                    return PromotedType{0};
                }();
                constexpr PromotedType rel_tol = [] {
                    if constexpr (std::is_same_v<T, double> || std::is_same_v<R, double>) return static_cast<PromotedType>(1e-12);
                    else if constexpr (std::is_same_v<T, float> || std::is_same_v<R, float>) return static_cast<PromotedType>(1e-6);
                    else if constexpr (std::is_same_v<T, float16> || std::is_same_v<R, float16>) return static_cast<PromotedType>(1e-3);
                    else if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<R, bfloat16>) return static_cast<PromotedType>(1e-2);
                    return PromotedType{0};
                }();
                bool is_close = (std::abs(val_a - val_b) <= std::max(rel_tol * std::max(std::abs(val_a), std::abs(val_b)), abs_tol));
                if constexpr (Op == CmpOp::Equal)
                    res[i] = is_close ? 1 : 0;
                else
                    res[i] = is_close ? 0 : 1;  // NotEqual
            } else {
                // 对于其他关系运算符，直接比较提升后的浮点数
                if constexpr (Op == CmpOp::Greater)
                    res[i] = (val_a > val_b);
                else if constexpr (Op == CmpOp::GreaterEqual)
                    res[i] = (val_a >= val_b);
                else if constexpr (Op == CmpOp::Less)
                    res[i] = (val_a < val_b);
                else if constexpr (Op == CmpOp::LessEqual)
                    res[i] = (val_a <= val_b);
            }
        }
    }
}
// --- 包装函数，提供清晰的 API ---
template <typename T, typename R>
void equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::Equal>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void not_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::NotEqual>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void greater_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::Greater>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void greater_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::GreaterEqual>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void less_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::Less>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void less_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, int size) {
    comparison_kernel<CmpOp::LessEqual>(res, a_ptr, b_ptr, size);
}

template <typename T>
size_t not_zero_kernel(const T *ptr, size_t n) {
    if (n == 0)
        return 0;
    if constexpr (std::is_integral_v<T>) {
        // 路径1: 整数类型。直接比较，最快。
        return std::count_if(std::execution::par, ptr, ptr + n,
                             [](T x) { return x != 0; });
    } else {
        // 1. 获取用于计算的类型
        using ComputeType = compute_type_t<T>;
        constexpr ComputeType tolerance = [] {
            if constexpr (std::is_same_v<T, bfloat16>)
                return static_cast<ComputeType>(1e-2);
            else if constexpr (std::is_same_v<T, float16>)
                return static_cast<ComputeType>(1e-3);
            else if constexpr (std::is_same_v<T, float>)
                return static_cast<ComputeType>(1e-8);
            else if constexpr (std::is_same_v<T, double>)
                return static_cast<ComputeType>(1e-12);
            return ComputeType{0};
        }();
        return std::count_if(std::execution::par, ptr, ptr + n,
                             [tolerance](T x) {
                                 return std::abs(static_cast<ComputeType>(x)) > tolerance;
                             });
    }
}

Tensor EqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        equal_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

Tensor LessImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        less_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

Tensor GreaterImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        greater_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

Tensor LessEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        less_equal_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

Tensor GreaterEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        greater_equal_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

Tensor NotEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    auto B = data_as_const_variant(b.dtype(), b.data());
    Tensor res(a.shape(), DataType::INT8, Device::CPU);
    std::visit([&](auto ptr_A, auto ptr_B) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        using BType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_B)>>;
        not_equal_kernel<AType, BType>(static_cast<int8_t *>(res.data()), ptr_A, ptr_B, res.numel());
    },
               A, B);
    return res;
}

size_t NonZeroImpl<Device::CPU>::execute(const Tensor &a) {
    switch (a.dtype()) {
        case DataType::INT8:
            return not_zero_kernel<int8_t>(static_cast<const int8_t *>(a.data()), a.numel());
        case DataType::INT16:
            return not_zero_kernel<int16_t>(static_cast<const int16_t *>(a.data()), a.numel());
        case DataType::INT32:
            return not_zero_kernel<int32_t>(static_cast<const int32_t *>(a.data()), a.numel());
        case DataType::INT64:
            return not_zero_kernel<int64_t>(static_cast<const int64_t *>(a.data()), a.numel());
        case DataType::FLOAT16:
            return not_zero_kernel<float16>(static_cast<const float16 *>(a.data()), a.numel());
        case DataType::BFLOAT16:
            return not_zero_kernel<bfloat16>(static_cast<const bfloat16 *>(a.data()), a.numel());
        case DataType::FLOAT32:
            return not_zero_kernel<float32>(static_cast<const float32 *>(a.data()), a.numel());
        case DataType::FLOAT64:
            return not_zero_kernel<float64>(static_cast<const float64 *>(a.data()), a.numel());
        default:
            throw std::runtime_error("mean: unsupported data type");
    }
}

template struct EqualImpl<Device::CPU>;
template struct LessImpl<Device::CPU>;
template struct GreaterImpl<Device::CPU>;
template struct LessEqualImpl<Device::CPU>;
template struct GreaterEqualImpl<Device::CPU>;
template struct NotEqualImpl<Device::CPU>;
template struct NonZeroImpl<Device::CPU>;


}  // namespace ops