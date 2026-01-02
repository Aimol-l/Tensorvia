#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

    template <via::Device D>
    struct SliceImpl;

    template <>
    struct SliceImpl<via::Device::CPU> {
        static Tensor execute(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);
    };

extern template struct SliceImpl<via::Device::CPU>;
}