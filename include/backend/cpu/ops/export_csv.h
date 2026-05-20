
#include "backend/cpu/cpu_tensor.h"
#include "ops.h"

namespace ops{

    template <via::Device D>
    struct ExportCsvImpl;

    template <>
    struct ExportCsvImpl<via::Device::CPU> {
        static void execute(const Tensor& a, const std::string& path);
    };

    extern template struct ExportCsvImpl<via::Device::CPU>;
}
