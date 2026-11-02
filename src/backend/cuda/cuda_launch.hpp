// CUDA 宿主驱动与内核之间的强类型启动契约。
//
// 旧的纯 C island 为每个内核暴露一个 C 链接入口，并把一切经 int dtype id 与
// void* 句柄打包传递。在 header-first 的 CUDA 产物中，宿主驱动与内核编译在同一
// 实体宇宙里，因此边界可以直接用库自己的类型：DataType 替代 id，cudaStream_t
// 替代 void*，cudaError_t 替代 int 状态码。不再有需要同步的 ABI。
#ifndef TVV_CUDA_LAUNCH_HPP
#define TVV_CUDA_LAUNCH_HPP

#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>

#include <cuda_runtime.h>

#include "tensorvia/backend/cpu/cpu_ops.hpp"  // ops::Conv2dShape：卷积的几何契约
#include "tensorvia/tensor.hpp"
#include "tensorvia/types.hpp"

namespace tvv::cuda {

// ---------------------------------------------------------------------------
// 宿主驱动与内核 switch 共享的算子码。
// ---------------------------------------------------------------------------

// 二元逐元素算子码（标量与张量-张量形式）。
inline constexpr int kBinAdd = 0;
inline constexpr int kBinSub = 1;
inline constexpr int kBinMul = 2;
inline constexpr int kBinDiv = 3;

// 一元逐元素算子码。p0/p1 的含义按每个算子码分别注明。
inline constexpr int kUnaryAbs     = 0;
inline constexpr int kUnarySin     = 1;
inline constexpr int kUnaryCos     = 2;
inline constexpr int kUnaryTan     = 3;
inline constexpr int kUnaryExp     = 4;
inline constexpr int kUnarySqrt    = 5;
inline constexpr int kUnaryLog     = 6;   // p0 = 对数底
inline constexpr int kUnaryPow     = 7;   // p0 = 指数
inline constexpr int kUnaryRelu    = 8;
inline constexpr int kUnarySilu    = 9;
inline constexpr int kUnaryTanh    = 10;
inline constexpr int kUnarySigmoid = 11;
inline constexpr int kUnaryClamp   = 12;  // p0 = 下界，p1 = 上界
inline constexpr int kUnaryErf     = 13;

// 归约算子码。
inline constexpr int kReduceSum   = 0;
inline constexpr int kReduceMin   = 1;
inline constexpr int kReduceMax   = 2;
inline constexpr int kReduceMean  = 3;
inline constexpr int kReduceProd  = 4;

// 比较算子码；结果是存放 0/1 的 INT8 张量。
inline constexpr int kCmpEqual        = 0;
inline constexpr int kCmpNotEqual     = 1;
inline constexpr int kCmpGreater      = 2;
inline constexpr int kCmpLess         = 3;
inline constexpr int kCmpGreaterEqual = 4;
inline constexpr int kCmpLessEqual    = 5;

inline constexpr int kMaxDims = kMaxTensorDims;

// ---------------------------------------------------------------------------
// 参数结构体。所有内容按值传给内核，不涉及托管内存暂存。
// ---------------------------------------------------------------------------

// 二元逐元素与比较内核的广播描述符。扁平输出下标用 `shape` 解码，再分别用各自的
// 步长数组累加两侧偏移（广播维的步长为 0）。
struct Bcast {
    int64_t numel = 0;
    int32_t ndim  = 0;
    int32_t reserved = 0;
    int64_t shape[kMaxDims] = {};
    int64_t sa[kMaxDims]    = {};
    int64_t sb[kMaxDims]    = {};
    int64_t off_a = 0;
    int64_t off_b = 0;
    int64_t off_o = 0;
};

// 单轴归约的几何形状：输入视为 outer x axis x inner。
struct ReduceAxis {
    int64_t outer = 0;
    int64_t axis  = 0;
    int64_t inner = 0;
};

// sort / topk 的几何：输入视为 outer x n x inner，每个 (outer, inner) 列独立
// 排序，输出排好序的前 keep 个（sort 的 keep = n，topk 的 keep = k）。
// `padded` 是 n 补齐到的 2 的幂（block 路径的 bitonic 网络用）。
struct SortAxis {
    int64_t outer     = 0;
    int64_t n         = 0;
    int64_t inner     = 0;
    int64_t keep      = 0;
    int64_t padded    = 0;
    int32_t descending = 0;  // 1 = 降序（topk 的 largest）
    int32_t reserved  = 0;
};

// 跨步视图重排（clone_as_contiguous）。
struct Repack {
    int64_t numel     = 0;
    int64_t offset    = 0;  // 视图起点，以元素计
    int32_t ndim      = 0;
    int32_t dtype_size = 0;
    int64_t shape[kMaxDims]   = {};
    int64_t strides[kMaxDims] = {};
};

// 跨步输入的切片。
struct Slice {
    int64_t numel  = 0;
    int64_t offset = 0;  // 输入视图起点，以元素计
    int32_t ndim   = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]   = {};
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t starts[kMaxDims]     = {};
};

// 把跨步输入按任意置换重排为连续输出。`axes[i]` 是成为输出第 i 维的输入维。
struct Transpose {
    int64_t numel = 0;
    int32_t ndim  = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]    = {};
    int64_t in_strides[kMaxDims]  = {};
    int64_t out_strides[kMaxDims] = {};
    int64_t axes[kMaxDims]        = {};
};

// 三路广播选择（where）：cond / a / b 各一套广播步长，各自叠加视图偏移。
struct Select {
    int64_t numel    = 0;
    int32_t ndim     = 0;
    int32_t cond_dtype = 0;  // DataType 的序号；cond 的标量类型在核心里查表
    int64_t shape[kMaxDims] = {};
    int64_t sc[kMaxDims]    = {};
    int64_t sa[kMaxDims]    = {};
    int64_t sb[kMaxDims]    = {};
    int64_t off_c = 0;
    int64_t off_a = 0;
    int64_t off_b = 0;
    int64_t off_o = 0;
};

// index_select 与 cumsum 的几何：输入视为 outer x axis_len x inner。
// `idx_len` 只对 index_select 有意义。
struct IndexAxis {
    int64_t outer    = 0;
    int64_t axis_len = 0;
    int64_t inner    = 0;
    int64_t idx_len  = 0;
};

// gather：输出形状与 index 相同，非 dim 维坐标决定输入 base。
struct Gather {
    int64_t numel       = 0;
    int32_t ndim        = 0;
    int32_t idx_dtype   = 0;  // DataType 的序号；index 的标量类型在核心里查表
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t axis_dim    = 0;  // dim 维长度，用于越界判定
    int64_t axis_stride = 0;  // dim 维在输入中的步长
    int32_t dim         = 0;
    int32_t reserved    = 0;
};

// ---------------------------------------------------------------------------
// 错误处理。
//
// 每次启动返回待处理错误（cudaGetLastError）而不做同步；驱动抛出时带上算子名，
// 使消息指向正确的栈帧。内部的“参数非法”结果使用对应的 cudaError_t 值，因此一次
// 检查即可覆盖两种情况。
// ---------------------------------------------------------------------------

inline void check_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::format("CUDA error in {}: {} ({})", what,
                                         cudaGetErrorString(err),
                                         cudaGetErrorName(err)));
}

// ---------------------------------------------------------------------------
// 启动入口。宿主驱动与内核是同一产物中的普通函数；启动不同步。
// ---------------------------------------------------------------------------

cudaError_t launch_fill(cudaStream_t stream, void* out, int64_t numel,
                        DataType dtype, double value);
cudaError_t launch_random(cudaStream_t stream, void* out, int64_t numel,
                          DataType dtype, float lo, float hi, uint64_t seed);

// 原地标量算术：buf = buf <op> scalar。
cudaError_t launch_scalar(cudaStream_t stream, void* buf, int64_t numel,
                          DataType dtype, int op, double scalar);

// 广播二元算术，写入 `out`。
cudaError_t launch_binary(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype_a, DataType dtype_b,
                          DataType dtype_out, int op, const Bcast& p);

// 对 `numel` 个连续元素做一元逐元素运算。dtype_in 与 dtype_out 仅在其它后端对
// 超越算子施加的整型 -> FLOAT32 提升时不同；计算精度跟随 dtype_in。
cudaError_t launch_unary(cudaStream_t stream, const void* in, void* out,
                         int64_t numel, DataType in_dtype, DataType out_dtype,
                         int op, double p0, double p1);

// 广播比较，输出 INT8。Equal/NotEqual 使用宿主按*原始*操作数 dtype 推导的容差；
// 大小比较是精确的，忽略容差。
cudaError_t launch_compare(cudaStream_t stream, const void* a, const void* b,
                           void* out, DataType dtype_a, DataType dtype_b, int op,
                           double tol_abs, double tol_rel, const Bcast& p);

// 在连续输入上广播的标量比较。
cudaError_t launch_compare_scalar(cudaStream_t stream, const void* a, void* out,
                                  int64_t numel, DataType dtype, int op,
                                  double scalar, double tol_abs, double tol_rel);

cudaError_t launch_matmul(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype, int64_t batch, int64_t rows,
                          int64_t cols, int64_t k);

cudaError_t launch_softmax(cudaStream_t stream, const void* in, void* out,
                           DataType in_dtype, DataType out_dtype,
                           const ReduceAxis& p);

// 单轴归约：`out` 为 outer * inner 个 out_dtype 元素。
cudaError_t launch_reduce_axis(cudaStream_t stream, const void* in, void* out,
                               DataType in_dtype, DataType out_dtype, int op,
                               const ReduceAxis& p);

// 沿轴求 argmax/argmin；`out` 为 outer * inner 个 INT32 元素。
cudaError_t launch_arg_reduce(cudaStream_t stream, const void* in, void* out,
                              DataType dtype, bool is_max, const ReduceAxis& p);

// sort / topk：每列独立排序后取前 keep 个。`out_vals` 与 `out_idx` 都是
// outer * keep * inner 个元素（vals 为 dtype，idx 为 INT64）。
cudaError_t launch_sort(cudaStream_t stream, const void* in, void* out_vals,
                        void* out_idx, DataType dtype, const SortAxis& p);

// All/Any：`out` 是一个 INT32 设备元素；宿主再把它拷回。
cudaError_t launch_all_any(cudaStream_t stream, const void* in, void* out,
                           int64_t numel, DataType dtype, bool is_all, double value);

// 非零元素个数；结果是一个 INT64 设备元素。
cudaError_t launch_nonzero_count(cudaStream_t stream, const void* in, void* out,
                                 int64_t numel, DataType dtype);

cudaError_t launch_cast(cudaStream_t stream, const void* in, void* out,
                        int64_t numel, DataType in_dtype, DataType out_dtype);

cudaError_t launch_concat(cudaStream_t stream, void* out,
                          const void* const* inputs, const int64_t* input_offsets,
                          int32_t num_inputs, DataType dtype,
                          const int64_t* out_strides, const int64_t* in_shapes,
                          const int64_t* in_strides, int32_t ndim, int32_t axis);

cudaError_t launch_slice(cudaStream_t stream, const void* in, void* out,
                         DataType dtype, const Slice& p);

cudaError_t launch_transpose(cudaStream_t stream, const void* in, void* out,
                             DataType dtype, const Transpose& p);

// 把跨步视图（由 `meta` 描述）物化为连续 buffer。
cudaError_t launch_repack(cudaStream_t stream, const void* in, void* out,
                          const Metadata& meta);

// where：cond 判真后从 a 或 b 取元素，写出 dtype 由 val_dtype 决定（BOOL 会规范化）。
cudaError_t launch_select(cudaStream_t stream, const void* cond, const void* a,
                          const void* b, void* out, DataType val_dtype, const Select& p);

// index_select / gather：`flag` 是设备上的 int32 越界标志（可为 nullptr），
// 任一索引落在 [0, axis_dim) 之外时置 1；核心里对越界索引做截断，因此即使调用方
// 不读标志也不会越界读写。
cudaError_t launch_index_select(cudaStream_t stream, const void* in, const void* idx,
                                void* out, DataType val_dtype, DataType idx_dtype,
                                const IndexAxis& p, int32_t* flag);

cudaError_t launch_gather(cudaStream_t stream, const void* in, const void* idx,
                          void* out, DataType val_dtype, DataType idx_dtype,
                          const Gather& p, int32_t* flag);

// conv2d：输入/输出都是连续 NCHW；`bias` 可为 nullptr（表示没有偏置）。
cudaError_t launch_conv2d(cudaStream_t stream, const void* in, const void* weight,
                          const void* bias, void* out, DataType dtype,
                          const ops::Conv2dShape& p);

// 沿 axis 的累积和；输出与输入同形同 dtype。
cudaError_t launch_cumsum(cudaStream_t stream, const void* in, void* out,
                          DataType dtype, const IndexAxis& p);

}  // namespace tvv::cuda

#endif  // TVV_CUDA_LAUNCH_HPP
