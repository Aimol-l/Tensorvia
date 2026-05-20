#include "backend/cpu/ops/export_csv.h"

#include <fstream>
#include <stdexcept>

#include "ops.h"
using namespace via;

namespace ops {

template <typename T>
inline void _export_csv(const Tensor& a, const std::string& path) {
    const auto& shape = a.shape();
    const T* data = static_cast<const T*>(a.data());

    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::runtime_error(std::format("export_csv: cannot open file '{}'", path));

    if (shape.size() == 1) {
        int64_t cols = shape[0];
        for (int64_t j = 0; j < cols; ++j) {
            if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
                ofs << static_cast<float>(data[j]);
            } else {
                ofs << data[j];
            }
            if (j != cols - 1) ofs << ',';
        }
        ofs << '\n';
    } else {
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
                    ofs << static_cast<float>(data[i * cols + j]);
                } else {
                    ofs << data[i * cols + j];
                }
                if (j != cols - 1) ofs << ',';
            }
            ofs << '\n';
        }
    }
}

void ExportCsvImpl<Device::CPU>::execute(const Tensor& a, const std::string& path) {
    const auto& shape = a.shape();
    if (shape.size() != 1 && shape.size() != 2)
        throw std::runtime_error(std::format("export_csv: only 1D or 2D tensors are supported, got {}D", shape.size()));

    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        _export_csv<T>(a, path);
    });
}

template struct ExportCsvImpl<Device::CPU>;
}  // namespace ops
