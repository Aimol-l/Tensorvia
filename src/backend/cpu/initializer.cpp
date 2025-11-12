#include "backend/cpu/ops/initializer.h"

#include <omp.h>

#include <cstddef>
#include <cstring>
#include <print>
#include <random>
#include <stdfloat>

namespace ops {

template <typename T>
inline void fill_value_impl(T* ptr, float val, size_t numel) {
    std::fill_n(static_cast<T*>(ptr), numel, T(val));
}
template <typename T>
void fill_random_impl(T* typed_ptr, size_t numel, float min, float max) {
    #pragma omp parallel
    {
        std::mt19937 rng(static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count() + omp_get_thread_num() * 9973)
        );
        #pragma omp for
        for (size_t i = 0; i < numel; ++i) {
            if constexpr (std::is_integral_v<T>) {
                std::uniform_int_distribution<T> dist(static_cast<T>(min), static_cast<T>(max));
                typed_ptr[i] = dist(rng);
            } else {
                std::uniform_real_distribution<float> dist(min, max);
                typed_ptr[i] = static_cast<T>(dist(rng));
            }
        }
    }
}

//****************************** 特化 ***********************************
Tensor ZerosImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, 0,res.numel());
    });
    return res;
}

Tensor OnesImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype) {
   Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, 1,res.numel());
    });
    return res;
}

Tensor FillImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float value) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_value_impl<T>(res_ptr, value,res.numel());
    });
    return res;
}

Tensor RandomImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max) {
    Tensor res(shape, dtype, Device::CPU);
    dispatch_dtype(dtype, [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* res_ptr = static_cast<T*>(res.data());
        fill_random_impl<T>(res_ptr,res.numel(),min,max);
    });
    return res;
}

template struct ZerosImpl<Device::CPU>;
template struct OnesImpl<Device::CPU>;
template struct FillImpl<Device::CPU>;
template struct RandomImpl<Device::CPU>;
}  // namespace ops