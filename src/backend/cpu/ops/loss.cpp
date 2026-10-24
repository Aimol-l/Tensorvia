#include <cmath>
#include <cstddef>
#include <cstdint>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"



namespace ops {

template <typename T>
float mse_kernel(const T* a, const T* b, size_t n) {
    float sse = 0.0f;
    #pragma omp parallel for simd reduction(+:sse) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sse += d * d;
    }
    return sse;
}

template <typename T>
float nll_kernel(const T* log_prob, const int64_t* target, size_t n, size_t c) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t i = 0; i < n; ++i)
        sum += static_cast<float>(log_prob[i * c + target[i]]);
    return sum;
}

float MseLossImpl<Device::CPU>::execute(const Tensor& input, const Tensor& target) {
    float sse = 0.0f;
    dispatch_dtype(input.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        sse = mse_kernel<T>(static_cast<const T*>(input.data()),
                            static_cast<const T*>(target.data()), input.numel());
    });
    return sse / static_cast<float>(input.numel());
}

float NllLossImpl<Device::CPU>::execute(const Tensor& input, const Tensor& target) {
    const size_t n = static_cast<size_t>(input.shape(0));
    const size_t c = static_cast<size_t>(input.shape(1));
    const int64_t* t = static_cast<const int64_t*>(target.data());
    float sum = 0.0f;
    dispatch_dtype(input.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        sum = nll_kernel<T>(static_cast<const T*>(input.data()), t, n, c);
    });
    return -sum / static_cast<float>(n);
}

}  // namespace ops
