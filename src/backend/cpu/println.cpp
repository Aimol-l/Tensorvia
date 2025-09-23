#include "backend/cpu/ops/println.h"

#include <cstddef>
#include <iostream>
#include <print>

#include "ops.h"

namespace ops {
template <typename T>
concept Float16Type = std::is_same_v<T, float16>;
template <typename T>
concept BFloat16Type = std::is_same_v<T, bfloat16>;

template <typename T>
inline void _println(const Tensor& a) {
    const std::vector<int64_t>& shape = a.shape();
    if (shape.empty()) {
        std::cout << "[ ]\n";
        return;
    }
    const T* data = static_cast<const T*>(a.data());
    constexpr size_t max_elements_per_dim = 7;
    size_t total_dims = shape.size();
    std::vector<size_t> strides(total_dims, 1);
    for (int i = total_dims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    auto print_recursive = [&](auto&& self, size_t dim, size_t offset, std::string indent) -> void {
        size_t dim_size = shape[dim];
        bool omit = dim_size > max_elements_per_dim;
        size_t show = omit ? 3 : dim_size;
        if (dim == total_dims - 1) {
            std::cout << std::format("{}[", indent);
            for (size_t i = 0; i < show; ++i) {
                if constexpr (Float16Type<T> || BFloat16Type<T>) {
                    std::cout << std::format("{:.3f}", float(data[offset + i]));
                } else if constexpr (std::is_integral_v<T>) {
                    std::cout << std::format("{}", data[offset + i]);
                } else {
                    std::cout << std::format("{:.3f}", data[offset + i]);
                }
                if (i != show - 1)
                    std::cout << ", ";
            }
            if (omit) {
                std::cout << ", ..., ";
                for (size_t i = dim_size - 3; i < dim_size; ++i) {
                    if constexpr (Float16Type<T> || BFloat16Type<T>) {
                        std::cout << std::format("{:.3f}", float(data[offset + i]));
                    } else if constexpr (std::is_integral_v<T>) {
                        std::cout << std::format("{}", data[offset + i]);
                    } else {
                        std::cout << std::format("{:.3f}", data[offset + i]);
                    }
                    if (i != dim_size - 1)
                        std::cout << ", ";
                }
            }
            std::cout << "]";
        } else {
            std::cout << std::format("{}[\n", indent);
            for (size_t i = 0; i < show; ++i) {
                self(self, dim + 1, offset + i * strides[dim], indent + "  ");
                std::cout << ",\n";
            }
            if (omit) {
                std::cout << std::format("{}  .....\n", indent);
                for (size_t i = dim_size - 3; i < dim_size; ++i) {
                    self(self, dim + 1, offset + i * strides[dim], indent + "  ");
                    if (i != dim_size - 1)
                        std::cout << ",\n";
                    else
                        std::cout << "\n";
                }
            }
            std::cout << std::format("{}]", indent);
        }
    };
    print_recursive(print_recursive, 0, 0, "");
    std::cout << std::format("\nTensor shape: (");
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << std::format("{}", shape[i]);
        if (i != shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << ") | ";
}

void PrintlnImpl<Device::CPU>::execute(const Tensor& a) {
    switch (a.dtype()) {
        case DataType::INT8:
            _println<int8_t>(a);
            break;
        case DataType::INT16:
            _println<int16_t>(a);
            break;
        case DataType::INT32:
            _println<int32_t>(a);
            break;
        case DataType::INT64:
            _println<int64_t>(a);
            break;
        case DataType::FLOAT16:
            _println<float16>(a);
            break;
        case DataType::FLOAT32:
            _println<float32>(a);
            break;
        case DataType::FLOAT64:
            _println<float64>(a);
            break;
        case DataType::BFLOAT16:
            _println<bfloat16>(a);
            break;
        default:
            std::cerr << "Unsupported dtype in println\n";
            break;
    }
}

template struct PrintlnImpl<Device::CPU>;
}  // namespace ops