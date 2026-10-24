#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

// Declarations only: this partition deliberately includes no Vulkan headers so
// it can be part of a CPU-only build. The definitions live in
// src/backend/vulkan/*.cpp, which is compiled only when BACKEND_VULKAN is on.
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"

namespace ops {

template <> struct ZerosImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct OnesImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct FillImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
    static void execute(Tensor& a, float value);
};
template <> struct RandomImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max);
};

template <> struct AddImpl<Device::VULKAN> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct SubImpl<Device::VULKAN> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct MulImpl<Device::VULKAN> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct DivImpl<Device::VULKAN> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};

template <> struct AbsImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SinImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct CosImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct ExpImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SqrtImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct LogImpl<Device::VULKAN> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct PowImpl<Device::VULKAN> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct ClampImpl<Device::VULKAN> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

template <> struct ReluImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SiluImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanhImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SigmoidImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SoftmaxImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, int axis);
};

template <> struct SumImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MinImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MaxImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MeanImpl<Device::VULKAN> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMaxImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMinImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct AllImpl<Device::VULKAN> {
    static bool execute(const Tensor& a, float value);
};
template <> struct AnyImpl<Device::VULKAN> {
    static bool execute(const Tensor& a, float value);
};

template <> struct EqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NotEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessEqualImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NonZeroImpl<Device::VULKAN> {
    static size_t execute(const Tensor& a);
};

template <> struct MatMulImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};

template <> struct TypecastImpl<Device::VULKAN> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

template <> struct ConcatImpl<Device::VULKAN> {
    static Tensor execute(const std::vector<Tensor>& tensors, int dim);
};

template <> struct TransposeImpl<Device::VULKAN> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);
};

template <> struct SliceImpl<Device::VULKAN> {
    static Tensor execute(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);
};

}  // namespace ops
