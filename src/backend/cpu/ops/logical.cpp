#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"



namespace ops {
template <OpType Op, typename T, typename R>
void comparison_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
#pragma omp parallel for if (size > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_integral_v<T> && std::is_integral_v<R>) {
            // 对整数直接进行比较，速度最快
            const auto val_a = a_ptr[i];
            const auto val_b = b_ptr[i];
            if constexpr (Op == OpType::Equal)
                res[i] = (val_a == val_b);
            else if constexpr (Op == OpType::NotEqual)
                res[i] = (val_a != val_b);
            else if constexpr (Op == OpType::Greater)
                res[i] = (val_a > val_b);
            else if constexpr (Op == OpType::GreaterEqual)
                res[i] = (val_a >= val_b);
            else if constexpr (Op == OpType::Less)
                res[i] = (val_a < val_b);
            else if constexpr (Op == OpType::LessEqual)
                res[i] = (val_a <= val_b);

        } else {
            using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
            const PromotedType val_a = a_ptr[i];
            const PromotedType val_b = b_ptr[i];
            if constexpr (Op == OpType::Equal || Op == OpType::NotEqual) {
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
                if constexpr (Op == OpType::Equal)
                    res[i] = is_close ? 1 : 0;
                else
                    res[i] = is_close ? 0 : 1;  // NotEqual
            } else {
                // 对于其他关系运算符，直接比较提升后的浮点数
                if constexpr (Op == OpType::Greater)
                    res[i] = (val_a > val_b);
                else if constexpr (Op == OpType::GreaterEqual)
                    res[i] = (val_a >= val_b);
                else if constexpr (Op == OpType::Less)
                    res[i] = (val_a < val_b);
                else if constexpr (Op == OpType::LessEqual)
                    res[i] = (val_a <= val_b);
            }
        }
    }
}
// 标量比较 kernel：b 侧为 float 常量。语义等价于 Fill(FLOAT32) 全形状
// 临时张量 + comparison_kernel<T, float32>（R 恒为浮点，无双整型快路径），
// 但省掉临时张量的分配、填充与整趟读取。
template <OpType Op, typename T>
void comparison_scalar_kernel(int8_t *res, const T *a_ptr, float scalar, size_t size) {
    using R = float32;
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    #pragma omp parallel for if (size > 4096)
    for (size_t i = 0; i < size; ++i) {
        const PromotedType val_a = a_ptr[i];
        const PromotedType val_b = scalar;
        if constexpr (Op == OpType::Equal || Op == OpType::NotEqual) {
            constexpr PromotedType abs_tol = [] {
                if constexpr (std::is_same_v<T, double>) return static_cast<PromotedType>(1e-9);
                else if constexpr (std::is_same_v<T, float>) return static_cast<PromotedType>(1e-5);
                else if constexpr (std::is_same_v<T, float16>) return static_cast<PromotedType>(1e-3);
                else if constexpr (std::is_same_v<T, bfloat16>) return static_cast<PromotedType>(1e-2);
                else return static_cast<PromotedType>(1e-5);  // 整型 vs float32 按 float 容差
            }();
            constexpr PromotedType rel_tol = [] {
                if constexpr (std::is_same_v<T, double>) return static_cast<PromotedType>(1e-12);
                else if constexpr (std::is_same_v<T, float>) return static_cast<PromotedType>(1e-6);
                else if constexpr (std::is_same_v<T, float16>) return static_cast<PromotedType>(1e-3);
                else if constexpr (std::is_same_v<T, bfloat16>) return static_cast<PromotedType>(1e-2);
                else return static_cast<PromotedType>(1e-6);
            }();
            const bool is_close = (std::abs(val_a - val_b) <=
                std::max(rel_tol * std::max(std::abs(val_a), std::abs(val_b)), abs_tol));
            if constexpr (Op == OpType::Equal)
                res[i] = is_close ? 1 : 0;
            else
                res[i] = is_close ? 0 : 1;
        } else {
            if constexpr (Op == OpType::Greater)
                res[i] = (val_a > val_b);
            else if constexpr (Op == OpType::GreaterEqual)
                res[i] = (val_a >= val_b);
            else if constexpr (Op == OpType::Less)
                res[i] = (val_a < val_b);
            else
                res[i] = (val_a <= val_b);
        }
    }
}

// --- 包装函数，提供清晰的 API ---
template <typename T, typename R>
void equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::Equal>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void not_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::NotEqual>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void greater_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::Greater>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void greater_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::GreaterEqual>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void less_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::Less>(res, a_ptr, b_ptr, size);
}

template <typename T, typename R>
void less_equal_kernel(int8_t *res, const T *a_ptr, const R *b_ptr, size_t size) {
    comparison_kernel<OpType::LessEqual>(res, a_ptr, b_ptr, size);
}

template <typename T>
size_t not_zero_kernel(const T *ptr, size_t n) {
    if (n == 0)
        return 0;
    if constexpr (std::is_integral_v<T>) {
        // 路径1: 整数类型。直接比较，最快。
        size_t count = 0;
        #pragma omp parallel for reduction(+:count) if (n > 4096)
        for (size_t i = 0; i < n; ++i) {
            if (ptr[i] != 0) {
                count++;
            }
        }
        return count;
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
        // 路径2: 浮点类型。使用容差比较。
        size_t count = 0;
        #pragma omp parallel for reduction(+:count) if (n > 4096)
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(static_cast<ComputeType>(ptr[i])) > tolerance) {
                count++;
            }
        }
        return count;
    }
}

// ================================================================
// ufunc Op：比较 kernel 保持混合 dtype 双类型分发，广播/物化由框架处理
struct EqualOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        equal_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::Equal, T>(out, a, val, n);
    }
};
struct NotEqualOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        not_equal_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::NotEqual, T>(out, a, val, n);
    }
};
struct GreaterOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        greater_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::Greater, T>(out, a, val, n);
    }
};
struct LessOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        less_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::Less, T>(out, a, val, n);
    }
};
struct GreaterEqualOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        greater_equal_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::GreaterEqual, T>(out, a, val, n);
    }
};
struct LessEqualOp {
    template <class T, class R>
    static void kernel(int8_t* out, const T* a, const R* b, size_t n) {
        less_equal_kernel<T, R>(out, a, b, n);
    }
    template <class T>
    static void kernel_scalar(int8_t* out, const T* a, float val, size_t n) {
        comparison_scalar_kernel<OpType::LessEqual, T>(out, a, val, n);
    }
};

Tensor EqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<EqualOp>(a, b); }
Tensor NotEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<NotEqualOp>(a, b); }
Tensor GreaterImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<GreaterOp>(a, b); }
Tensor LessImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<LessOp>(a, b); }
Tensor GreaterEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<GreaterEqualOp>(a, b); }
Tensor LessEqualImpl<Device::CPU>::execute(const Tensor &a, const Tensor &b) { return ufunc::logical_op<LessEqualOp>(a, b); }

size_t NonZeroImpl<Device::CPU>::execute(const Tensor &a) {
    size_t count = 0;
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        const T* a_ptr = static_cast<const T*>(a.data());
        count = not_zero_kernel<T>(a_ptr,a.numel());
    });
    return count;
}

Tensor EqualImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<EqualOp>(a, val); }
Tensor NotEqualImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<NotEqualOp>(a, val); }
Tensor GreaterImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<GreaterOp>(a, val); }
Tensor LessImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<LessOp>(a, val); }
Tensor GreaterEqualImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<GreaterEqualOp>(a, val); }
Tensor LessEqualImpl<Device::CPU>::execute(const Tensor &a, float val) { return ufunc::logical_op_scalar<LessEqualOp>(a, val); }

}  // namespace ops
