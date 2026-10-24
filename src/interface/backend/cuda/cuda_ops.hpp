#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Declarations only: this partition includes no CUDA header and is compiled
// even in CPU-only builds. Definitions live in src/backend/cuda/*.cpp, which
// are part of the build only when BACKEND_CUDA is defined.
//
// Only operations whose island kernels exist are declared. An undeclared
// specialization makes execute_on's `has_cuda_impl` concept false, which turns
// into the "not implemented on the CUDA backend" runtime error instead of a
// link failure.
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"

namespace ops {

template <> struct ZerosImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct OnesImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct FillImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
    static void execute(Tensor& a, float value);
};
template <> struct RandomImpl<Device::CUDA> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max);
};

// ---------------------------------------------------------------------------
// Arithmetic.
// ---------------------------------------------------------------------------
template <> struct AddImpl<Device::CUDA> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct SubImpl<Device::CUDA> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct MulImpl<Device::CUDA> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct DivImpl<Device::CUDA> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};

template <> struct AbsImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SinImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct CosImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct ExpImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SqrtImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct PowImpl<Device::CUDA> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LogImpl<Device::CUDA> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct ClampImpl<Device::CUDA> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

// ---------------------------------------------------------------------------
// Activations.
// ---------------------------------------------------------------------------
template <> struct ReluImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SiluImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanhImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SigmoidImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SoftmaxImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};

// ---------------------------------------------------------------------------
// Reductions.
// ---------------------------------------------------------------------------
template <> struct SumImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MinImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MaxImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MeanImpl<Device::CUDA> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMaxImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMinImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct AllImpl<Device::CUDA> {
    static bool execute(const Tensor& a, float value);
};
template <> struct AnyImpl<Device::CUDA> {
    static bool execute(const Tensor& a, float value);
};
template <> struct NonZeroImpl<Device::CUDA> {
    static size_t execute(const Tensor& a);
};

// ---------------------------------------------------------------------------
// Dtype conversion.
// ---------------------------------------------------------------------------
template <> struct TypecastImpl<Device::CUDA> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

// ---------------------------------------------------------------------------
// Comparison and logical.
// ---------------------------------------------------------------------------
template <> struct EqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NotEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessEqualImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};

// ---------------------------------------------------------------------------
// Shape and data movement.
// ---------------------------------------------------------------------------
template <> struct ConcatImpl<Device::CUDA> {
    static Tensor execute(const std::vector<Tensor>& tensors, int dim);
};
template <> struct SliceImpl<Device::CUDA> {
    static Tensor execute(const Tensor& t,
                          const std::vector<std::pair<int64_t, int64_t>>& ranges);
};
template <> struct TransposeImpl<Device::CUDA> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);
};

// ---------------------------------------------------------------------------
// Matrix multiplication.
// ---------------------------------------------------------------------------
template <> struct MatMulImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};

}  // namespace ops
