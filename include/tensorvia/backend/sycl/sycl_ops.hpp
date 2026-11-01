#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// 只有声明：这个分区不包含任何 SYCL 头，纯 CPU 构建里也会编译。实现体在
// src/backend/sycl/host/*.cpp，只有定义 BACKEND_SYCL 时才加入构建。
//
// 只声明设备内核已存在的算子。未声明的特化会让 execute_on 的 `has_sycl_impl`
// 概念为假，从而得到 "not implemented on the SYCL backend" 运行期错误，
// 而不是链接失败。
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/cpu_ops.hpp"

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
// 算术。
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
// 激活。
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
// 归约。
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
// dtype 转换。
// ---------------------------------------------------------------------------
template <> struct TypecastImpl<Device::SYCL> {
    static void execute(Tensor& a, DataType dst_type);
    static Tensor execute(const Tensor& a, DataType dst_type);
};

// ---------------------------------------------------------------------------
// 比较与逻辑。
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
// 形状与数据搬移。
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
// 矩阵乘。
// ---------------------------------------------------------------------------
template <> struct MatMulImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, const Tensor& b);
};
template <> struct Conv2dImpl<Device::SYCL> {
    static Tensor execute(const Tensor& input, const Tensor& weight, const Tensor* bias,
                          const Conv2dShape& shape);
};

// ---------------------------------------------------------------------------
// 条件选择、索引与累积和。
// ---------------------------------------------------------------------------
template <> struct WhereImpl<Device::SYCL> {
    static Tensor execute(const Tensor& cond, const Tensor& a, const Tensor& b);
    static Tensor execute(const Tensor& cond, const Tensor& a, float b);
    static Tensor execute(const Tensor& cond, float a, const Tensor& b);
};
template <> struct IndexSelectImpl<Device::SYCL> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};
template <> struct GatherImpl<Device::SYCL> {
    static Tensor execute(const Tensor& input, int dim, const Tensor& index);
};
template <> struct CumsumImpl<Device::SYCL> {
    static Tensor execute(const Tensor& a, int axis);
};

}  // namespace ops
