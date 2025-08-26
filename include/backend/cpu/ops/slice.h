#pragma once
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops {

    template <Device D>
    struct SliceImpl;

    template <>
    struct SliceImpl<Device::CPU> {
        static Tensor execute(const Tensor& t, const std::vector<std::pair<int, int>>& ranges);
    };

extern template struct SliceImpl<Device::CPU>;
}