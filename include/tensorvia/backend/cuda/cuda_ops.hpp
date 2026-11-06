#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"

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
// 算术。
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
template <> struct ErfImpl<Device::CUDA> {
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
// 激活。
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
template <> struct LayerNormImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& weight, const Tensor& bias,
                          int axis, float eps);
};
template <> struct RmsNormImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& weight, int axis, float eps);
};

// ---------------------------------------------------------------------------
// 归约。
// ---------------------------------------------------------------------------
template <> struct SumImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MinImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MaxImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct MeanImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};
template <> struct ProdImpl<Device::CUDA> {
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
// sort / topk。轴长上限 1024（超过显式抛错，见 sort.cu 的注释）。
// ---------------------------------------------------------------------------
template <> struct SortImpl<Device::CUDA> {
    static SortResult execute(const Tensor& input, int dim, bool descending);
};
template <> struct TopkImpl<Device::CUDA> {
    static TopkResult execute(const Tensor& input, int64_t k, int dim, bool largest);
};

// ---------------------------------------------------------------------------
// dtype 转换。
// ---------------------------------------------------------------------------
template <> struct TypecastImpl<Device::CUDA> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

// ---------------------------------------------------------------------------
// 比较与逻辑。
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
// 形状与数据搬移。
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
// 矩阵乘。
// ---------------------------------------------------------------------------
template <> struct MatMulImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <> struct Conv2dImpl<Device::CUDA> {
    static Tensor execute(const Tensor& input, const Tensor& weight, const Tensor* bias,
                          const Conv2dShape& shape);
};
template <> struct MaxPool2dImpl<Device::CUDA> {
    static Tensor execute(const Tensor& input, const Pool2dShape& shape);
};
template <> struct AvgPool2dImpl<Device::CUDA> {
    static Tensor execute(const Tensor& input, const Pool2dShape& shape);
};

// ---------------------------------------------------------------------------
// 条件选择、索引与累积和。
// ---------------------------------------------------------------------------
template <> struct WhereImpl<Device::CUDA> {
    static Tensor execute(const Tensor& cond, const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& cond, const Tensor& a, float b);
    static Tensor execute(const Tensor& cond, float a, const Tensor& b);
};
template <> struct IndexSelectImpl<Device::CUDA> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};
template <> struct GatherImpl<Device::CUDA> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};
template <> struct CumsumImpl<Device::CUDA> {
    static Tensor execute(const Tensor& a, int axis);
};

}  // namespace ops
