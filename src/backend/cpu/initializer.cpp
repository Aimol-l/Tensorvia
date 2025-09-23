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
// T* typed_ptr = static_cast<T*>(ptr);
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
    Tensor temp(shape, dtype, Device::CPU);
    void* data_ptr = temp.data();
    size_t numel = temp.numel();
    switch (dtype) {
        case DataType::INT8:
            fill_value_impl<int8_t>(static_cast<int8_t*>(data_ptr), 0, numel);
            break;
        case DataType::INT16:
            fill_value_impl<int16_t>(static_cast<int16_t*>(data_ptr), 0, numel);
            break;
        case DataType::INT32:
            fill_value_impl<int32_t>(static_cast<int32_t*>(data_ptr), 0, numel);
            break;
        case DataType::INT64:
            fill_value_impl<int64_t>(static_cast<int64_t*>(data_ptr), 0, numel);
            break;
        case DataType::FLOAT16:
            fill_value_impl<float16>(static_cast<float16*>(data_ptr), 0, numel);
            break;
        case DataType::FLOAT32:
            fill_value_impl<float32>(static_cast<float32*>(data_ptr), 0, numel);
            break;
        case DataType::FLOAT64:
            fill_value_impl<float64>(static_cast<float64*>(data_ptr), 0, numel);
            break;
        case DataType::BFLOAT16:
            fill_value_impl<bfloat16>(static_cast<bfloat16*>(data_ptr), 0, numel);
            break;
    }
    return temp;
}

Tensor OnesImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype) {
    Tensor temp(shape, dtype, Device::CPU);
    void* data_ptr = temp.data();
    size_t numel = temp.numel();
    switch (dtype) {
        case DataType::INT8:
            fill_value_impl<int8_t>(static_cast<int8_t*>(data_ptr), 1, numel);
            break;
        case DataType::INT16:
            fill_value_impl<int16_t>(static_cast<int16_t*>(data_ptr), 1, numel);
            break;
        case DataType::INT32:
            fill_value_impl<int32_t>(static_cast<int32_t*>(data_ptr), 1, numel);
            break;
        case DataType::INT64:
            fill_value_impl<int64_t>(static_cast<int64_t*>(data_ptr), 1, numel);
            break;
        case DataType::FLOAT16:
            fill_value_impl<float16>(static_cast<float16*>(data_ptr), 1, numel);
            break;
        case DataType::FLOAT32:
            fill_value_impl<float32>(static_cast<float32*>(data_ptr), 1, numel);
            break;
        case DataType::FLOAT64:
            fill_value_impl<float64>(static_cast<float64*>(data_ptr), 1, numel);
            break;
        case DataType::BFLOAT16:
            fill_value_impl<bfloat16>(static_cast<bfloat16*>(data_ptr), 1, numel);
            break;
    }
    return temp;
}

Tensor FillImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float value) {
    Tensor temp(shape, dtype, Device::CPU);
    void* data_ptr = temp.data();
    size_t numel = temp.numel();
    switch (dtype) {
        case DataType::INT8:
            fill_value_impl<int8_t>(static_cast<int8_t*>(data_ptr), value, numel);
            break;
        case DataType::INT16:
            fill_value_impl<int16_t>(static_cast<int16_t*>(data_ptr), value, numel);
            break;
        case DataType::INT32:
            fill_value_impl<int32_t>(static_cast<int32_t*>(data_ptr), value, numel);
            break;
        case DataType::INT64:
            fill_value_impl<int64_t>(static_cast<int64_t*>(data_ptr), value, numel);
            break;
        case DataType::FLOAT16:
            fill_value_impl<float16>(static_cast<float16*>(data_ptr), value, numel);
            break;
        case DataType::FLOAT32:
            fill_value_impl<float32>(static_cast<float32*>(data_ptr), value, numel);
            break;
        case DataType::FLOAT64:
            fill_value_impl<float64>(static_cast<float64*>(data_ptr), value, numel);
            break;
        case DataType::BFLOAT16:
            fill_value_impl<bfloat16>(static_cast<bfloat16*>(data_ptr), value, numel);
            break;
    }
    return temp;
}

Tensor RandomImpl<Device::CPU>::execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max) {
    Tensor temp(shape, dtype, Device::CPU);
    void* data_ptr = temp.data();
    size_t numel = temp.numel();
    switch (dtype) {
        case DataType::INT8:
            fill_random_impl<int8_t>(static_cast<int8_t*>(data_ptr), numel, min, max);
            break;
        case DataType::INT16:
            fill_random_impl<int16_t>(static_cast<int16_t*>(data_ptr), numel, min, max);
            break;
        case DataType::INT32:
            fill_random_impl<int32_t>(static_cast<int32_t*>(data_ptr), numel, min, max);
            break;
        case DataType::INT64:
            fill_random_impl<int64_t>(static_cast<int64_t*>(data_ptr), numel, min, max);
            break;
        case DataType::FLOAT16:
            fill_random_impl<float16>(static_cast<float16*>(data_ptr), numel, min, max);
            break;
        case DataType::FLOAT32:
            fill_random_impl<float32>(static_cast<float32*>(data_ptr), numel, min, max);
            break;
        case DataType::FLOAT64:
            fill_random_impl<float64>(static_cast<float64*>(data_ptr), numel, min, max);
            break;
        case DataType::BFLOAT16:
            fill_random_impl<bfloat16>(static_cast<bfloat16*>(data_ptr), numel, min, max);
            break;
        default:
            throw std::runtime_error("Unsupported dtype for random initializer.");
    }
    return temp;
}

template struct ZerosImpl<Device::CPU>;
template struct OnesImpl<Device::CPU>;
template struct FillImpl<Device::CPU>;
template struct RandomImpl<Device::CPU>;


}  // namespace ops