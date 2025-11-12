#include "backend/cpu/ops/activate.h"


#include <cmath>

namespace ops {

template <typename T>
void relu_kernel(const T* src, T* dst, size_t n) {
    #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] > T(0) ? src[i] : T(0);
    }
}

template <typename T, typename R>
void silu_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<R>(PromotedType(src[i]) / (1 + std::exp(-PromotedType(src[i]))));
    }
}
template <typename T, typename R = T>
void tanh_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<R>(std::tanh(PromotedType(src[i])));
    }
}

template <typename T, typename R>
void sigmoid_kernel(const T* src, R* dst, size_t n) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    #pragma omp parallel for if (n > 4096)  // 4096 是一个经验值
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<R>(1 / (1 + std::exp(-PromotedType(src[i]))));
    }
}
template <typename T, typename R>
void softmax_kernel(const T* src, R* res_ptr, size_t outer_size, size_t axis_size, size_t inner_size) {
    using PromotedType = decltype(std::declval<compute_type_t<T>>() + std::declval<compute_type_t<R>>());
    #pragma omp parallel for
    for (size_t outer_index = 0; outer_index < outer_size; ++outer_index) {
        if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float16>) {
            for (size_t inner_index = 0; inner_index < inner_size; ++inner_index) {
                R max_val = -std::numeric_limits<R>::infinity();
                for (size_t k = 0; k < axis_size; ++k) {
                    R curr_val = src[outer_index * axis_size * inner_size + k * inner_size + inner_index];
                    max_val = max_val > curr_val ? max_val : curr_val;
                }
                float exp_sum = 0;
                for (size_t k = 0; k < axis_size; ++k) {
                    R curr_val = static_cast<R>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index]) - max_val;
                    exp_sum += std::exp(float(curr_val));
                }
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t pos = outer_index * axis_size * inner_size + k * inner_size + inner_index;
                    res_ptr[pos] = R(std::exp(static_cast<float>(T(src[pos]) - T(max_val))) / exp_sum);
                }
            }
        } else {
            for (size_t inner_index = 0; inner_index < inner_size; ++inner_index) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t k = 0; k < axis_size; ++k) {
                    max_val = std::max(max_val, static_cast<float>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index]));
                }
                float exp_sum = 0;
                for (size_t k = 0; k < axis_size; ++k) {
                    exp_sum += std::exp(static_cast<float>(src[outer_index * axis_size * inner_size + k * inner_size + inner_index] - max_val));
                }
                for (size_t k = 0; k < axis_size; ++k) {
                    size_t pos = outer_index * axis_size * inner_size + k * inner_size + inner_index;
                    res_ptr[pos] = std::exp(static_cast<float>(src[pos] - max_val)) / exp_sum;
                }
            }
        }
    }
}


// ================================================================

void ReluImpl<Device::CPU>::execute(Tensor& a) {
    // auto A = data_as_const_variant(a.dtype(), a.data());
    // std::visit([&](auto ptr_A) {
    //     using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
    //     relu_kernel<AType>(ptr_A, static_cast<AType*>(a.data()), a.numel());
    // },A);
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* a_ptr = static_cast<T*>(a.data());
        relu_kernel<T>(a_ptr,a_ptr,a.numel());
    });
}
Tensor ReluImpl<Device::CPU>::execute(const Tensor& a) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    Tensor res = Tensor(a.shape(), a.dtype(), Device::CPU);
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;  // const T* --> const T --> T
        relu_kernel<AType>(ptr_A, static_cast<AType*>(res.data()), a.numel());
    },A);
    return res;
}

void SiluImpl<Device::CPU>::execute(Tensor& a) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        silu_kernel<AType>(static_cast<AType*>(a.data()), static_cast<AType*>(a.data()), a.numel());
    },
               A);
}
Tensor SiluImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;

    Tensor res = Tensor(a.shape(), res_type, Device::CPU);

    auto A = data_as_const_variant(a.dtype(), a.data());        // const T*
    auto Res = data_as_const_variant(res.dtype(), res.data());  // const R*

    std::visit([&](auto ptr_A, auto ptr_res) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;      // const T* --> const T --> T
        using ResType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_res)>>;  // const R* --> const R --> R
        silu_kernel<AType>(ptr_A, static_cast<ResType*>(res.data()), a.numel());
    },A, Res);

    return res;
}

void TanhImpl<Device::CPU>::execute(Tensor& a) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        tanh_kernel<AType>(static_cast<AType*>(a.data()), static_cast<AType*>(a.data()), a.numel());
    },
               A);
}
Tensor TanhImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;

    Tensor res = Tensor(a.shape(), res_type, Device::CPU);

    auto A = data_as_const_variant(a.dtype(), a.data());        // const T*
    auto Res = data_as_const_variant(res.dtype(), res.data());  // const R*

    std::visit([&](auto ptr_A, auto ptr_res) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;      // const T* --> const T --> T
        using ResType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_res)>>;  // const R* --> const R --> R
        tanh_kernel<AType>(ptr_A, static_cast<ResType*>(res.data()), a.numel());
    },
               A, Res);

    return res;
}

void SigmoidImpl<Device::CPU>::execute(Tensor& a) {
    auto A = data_as_const_variant(a.dtype(), a.data());
    std::visit([&](auto ptr_A) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;
        sigmoid_kernel<AType>(static_cast<AType*>(a.data()), static_cast<AType*>(a.data()), a.numel());
    },
               A);
}
Tensor SigmoidImpl<Device::CPU>::execute(const Tensor& a) {
    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;

    Tensor res = Tensor(a.shape(), res_type, Device::CPU);

    auto A = data_as_const_variant(a.dtype(), a.data());        // const T*
    auto Res = data_as_const_variant(res.dtype(), res.data());  // const R*

    std::visit([&](auto ptr_A, auto ptr_res) {
        using AType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_A)>>;      // const T* --> const T --> T
        using ResType = std::remove_cv_t<std::remove_pointer_t<decltype(ptr_res)>>;  // const R* --> const R --> R
        sigmoid_kernel<AType>(ptr_A, static_cast<ResType*>(res.data()), a.numel());
    },
               A, Res);

    return res;
}

Tensor SoftmaxImpl<Device::CPU>::execute(const Tensor& a, int axis) {
    size_t dim = a.shape().size();
    if (axis < 0)   axis += dim;
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i)
        outer_size *= a.shape(i);
    size_t axis_size = a.shape(axis);
    size_t inner_size = 1;
    for (int i = axis + 1; i < dim; ++i)
        inner_size *= a.shape(i);

    DataType res_type = a.dtype();
    if (a.dtype() <= DataType::INT64)
        res_type = DataType::FLOAT32;
    Tensor res(a.shape(), res_type, a.device());

    switch (a.dtype()) {
        case DataType::FLOAT64:
            softmax_kernel<float64>(static_cast<const float64*>(a.data()), static_cast<float64*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::FLOAT32:
            softmax_kernel<float32>(static_cast<const float32*>(a.data()), static_cast<float32*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::FLOAT16:
            softmax_kernel<float16>(static_cast<const float16*>(a.data()), static_cast<float16*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::BFLOAT16:
            softmax_kernel<bfloat16>(static_cast<const bfloat16*>(a.data()), static_cast<bfloat16*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::INT64:
            softmax_kernel<int64_t>(static_cast<const int64_t*>(a.data()), static_cast<float32*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::INT32:
            softmax_kernel<int32_t>(static_cast<const int32_t*>(a.data()), static_cast<float32*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::INT16:
            softmax_kernel<int16_t>(static_cast<const int16_t*>(a.data()), static_cast<float32*>(res.data()), outer_size, axis_size, inner_size);
            break;
        case DataType::INT8:
            softmax_kernel<int8_t>(static_cast<const int8_t*>(a.data()), static_cast<float32*>(res.data()), outer_size, axis_size, inner_size);
            break;
        default:
            std::runtime_error("Unsupported data type");
    }
    return res;
}

template struct ReluImpl<Device::CPU>;
template struct SiluImpl<Device::CPU>;
template struct TanhImpl<Device::CPU>;
template struct SigmoidImpl<Device::CPU>;
template struct SoftmaxImpl<Device::CPU>;


}  // namespace ops