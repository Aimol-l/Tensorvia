
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops{
    
    template <Device D>
    struct PrintlnImpl;

    template <>
    struct PrintlnImpl<Device::CPU> {
        static void execute(const Tensor& a);
    };

    extern template struct PrintlnImpl<Device::CPU>;
}