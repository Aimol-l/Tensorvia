// Device::CPU 的融合块 Conv2d+Relu+MaxPool2d(2x2)。
//
// CPU 侧就是组合式：三个算子在 CPU 上都不慢，融合内核省不下什么，组合式
// 还顺带保证了"与逐算子调用逐位一致"（测试的参考就是它）。设备后端的融合
// 内核省的是提交与同步往返，见各自实现。
#include <cstddef>
#include <cstdint>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"

namespace ops {

Tensor Conv2dReluMaxPool2dImpl<Device::CPU>::execute(const Tensor& input,
                                                     const Tensor& weight,
                                                     const Tensor* bias,
                                                     const ConvPoolShape& shape) {
    const Tensor conv = bias != nullptr
                            ? ops::Conv2d(input, weight, *bias, shape.conv.stride,
                                          shape.conv.padding)
                            : ops::Conv2d(input, weight, shape.conv.stride, shape.conv.padding);
    return ops::MaxPool2d(ops::Relu(conv), 2, 2);
}

}  // namespace ops
