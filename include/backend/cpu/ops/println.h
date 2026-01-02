
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops{
    
    template <via::Device D>
    struct PrintlnImpl;

    template <>
    struct PrintlnImpl<via::Device::CPU> {
        static void execute(const Tensor& a);
    };

    extern template struct PrintlnImpl<via::Device::CPU>;
}