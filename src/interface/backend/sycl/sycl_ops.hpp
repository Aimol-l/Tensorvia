#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// Declarations only: this partition includes no SYCL header and is compiled
// even in CPU-only builds. Definitions live in src/backend/sycl/host/*.cpp,
// which are part of the build only when BACKEND_SYCL is defined.
//
// Only operations whose device kernels exist are declared. An undeclared
// specialization makes execute_on's `has_sycl_impl` concept false, which turns
// into the "not implemented on the SYCL backend" runtime error instead of a
// link failure.
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/cpu_ops.hpp"

namespace ops {

template <> struct ZerosImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct OnesImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype);
};
template <> struct FillImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float value);
    static void execute(Tensor& a, float value);
};
template <> struct RandomImpl<Device::SYCL> {
    static Tensor execute(const std::vector<int64_t>& shape, DataType dtype, float min, float max);
};

// ---------------------------------------------------------------------------
// Arithmetic.
// ---------------------------------------------------------------------------
template <> struct AddImpl<Device::SYCL> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct SubImpl<Device::SYCL> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct MulImpl<Device::SYCL> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};
template <> struct DivImpl<Device::SYCL> {
    static void execute(Tensor& a, float b);
    static Tensor execute(const Tensor& a, float b);
    static Tensor execute(const Tensor& a, const Tensor& b);
    static void execute(const Tensor& a, const Tensor& b, Tensor& dst);
};

template <> struct AbsImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SinImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct CosImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct ExpImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SqrtImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct PowImpl<Device::SYCL> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LogImpl<Device::SYCL> {
    static void execute(Tensor& a, float val);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct ClampImpl<Device::SYCL> {
    static void execute(Tensor& a, float min, float max);
    static Tensor execute(const Tensor& a, float min, float max);
};

// ---------------------------------------------------------------------------
// Activations.
// ---------------------------------------------------------------------------
template <> struct ReluImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SiluImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct TanhImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SigmoidImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a);
};
template <> struct SoftmaxImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, int axis);
};

// ---------------------------------------------------------------------------
// Reductions.
// ---------------------------------------------------------------------------
template <> struct SumImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MinImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MaxImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MeanImpl<Device::SYCL> {
    static float execute(const Tensor& a);
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMaxImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ArgMinImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct AllImpl<Device::SYCL> {
    static bool execute(const Tensor& a, float value);
};
template <> struct AnyImpl<Device::SYCL> {
    static bool execute(const Tensor& a, float value);
};
template <> struct NonZeroImpl<Device::SYCL> {
    static size_t execute(const Tensor& a);
};

// ---------------------------------------------------------------------------
// Dtype conversion.
// ---------------------------------------------------------------------------
template <> struct TypecastImpl<Device::SYCL> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

// ---------------------------------------------------------------------------
// Comparison and logical.
// ---------------------------------------------------------------------------
template <> struct EqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct NotEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct GreaterEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};
template <> struct LessEqualImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& a, float val);
};

// ---------------------------------------------------------------------------
// Shape and data movement.
// ---------------------------------------------------------------------------
template <> struct ConcatImpl<Device::SYCL> {
    static Tensor execute(const std::vector<Tensor>& tensors, int dim);
};
template <> struct SliceImpl<Device::SYCL> {
    static Tensor execute(const Tensor& t,
                          const std::vector<std::pair<int64_t, int64_t>>& ranges);
};
template <> struct TransposeImpl<Device::SYCL> {
    static void execute(Tensor& a);
    static Tensor execute(const Tensor& a, std::initializer_list<int64_t> axes);
    static void execute(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);
};

// ---------------------------------------------------------------------------
// Matrix multiplication.
// ---------------------------------------------------------------------------
template <> struct MatMulImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};

}  // namespace ops
