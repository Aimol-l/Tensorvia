// Typed launch contract between the CUDA host drivers and the kernels.
//
// The old pure-C island exposed one C-linkage entry point per kernel and
// marshalled everything through int dtype ids and void* handles. In the
// header-first CUDA artifact the host driver and the kernels are compiled in
// the same entity universe, so the boundary can use the library's own types:
// DataType instead of an id, cudaStream_t instead of void*, cudaError_t instead
// of an int status. There is no ABI left to keep in sync.
#ifndef TVV_CUDA_LAUNCH_HPP
#define TVV_CUDA_LAUNCH_HPP

#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>

#include <cuda_runtime.h>

#include "interface/tensor.hpp"
#include "interface/types.hpp"

namespace tvv::cuda {

// ---------------------------------------------------------------------------
// Operator codes shared by the host drivers and the kernel switches.
// ---------------------------------------------------------------------------

// Binary element-wise codes (scalar and tensor-tensor forms).
inline constexpr int kBinAdd = 0;
inline constexpr int kBinSub = 1;
inline constexpr int kBinMul = 2;
inline constexpr int kBinDiv = 3;

// Unary element-wise codes. p0/p1 meaning is documented per code.
inline constexpr int kUnaryAbs     = 0;
inline constexpr int kUnarySin     = 1;
inline constexpr int kUnaryCos     = 2;
inline constexpr int kUnaryTan     = 3;
inline constexpr int kUnaryExp     = 4;
inline constexpr int kUnarySqrt    = 5;
inline constexpr int kUnaryLog     = 6;   // p0 = logarithm base
inline constexpr int kUnaryPow     = 7;   // p0 = exponent
inline constexpr int kUnaryRelu    = 8;
inline constexpr int kUnarySilu    = 9;
inline constexpr int kUnaryTanh    = 10;
inline constexpr int kUnarySigmoid = 11;
inline constexpr int kUnaryClamp   = 12;  // p0 = min, p1 = max

// Reduction codes.
inline constexpr int kReduceSum  = 0;
inline constexpr int kReduceMin  = 1;
inline constexpr int kReduceMax  = 2;
inline constexpr int kReduceMean = 3;

// Comparison codes; the result is an INT8 tensor holding 0 or 1.
inline constexpr int kCmpEqual        = 0;
inline constexpr int kCmpNotEqual     = 1;
inline constexpr int kCmpGreater      = 2;
inline constexpr int kCmpLess         = 3;
inline constexpr int kCmpGreaterEqual = 4;
inline constexpr int kCmpLessEqual    = 5;

inline constexpr int kMaxDims = kMaxTensorDims;

// ---------------------------------------------------------------------------
// Parameter structs. Everything is passed by value to the kernels, so no
// managed-memory staging is involved.
// ---------------------------------------------------------------------------

// Broadcast descriptor for binary element-wise and comparison kernels.
// A flat output index is decoded with `shape`, then each side's offset is
// accumulated with its own stride array (0 for broadcast dimensions).
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

// Axis reduction geometry: the input is viewed as outer x axis x inner.
struct ReduceAxis {
    int64_t outer = 0;
    int64_t axis  = 0;
    int64_t inner = 0;
};

// Strided view repack (clone_as_contiguous).
struct Repack {
    int64_t numel     = 0;
    int64_t offset    = 0;  // view origin, in elements
    int32_t ndim      = 0;
    int32_t dtype_size = 0;
    int64_t shape[kMaxDims]   = {};
    int64_t strides[kMaxDims] = {};
};

// Slice of a strided input.
struct Slice {
    int64_t numel  = 0;
    int64_t offset = 0;  // input view origin, in elements
    int32_t ndim   = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]   = {};
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t starts[kMaxDims]     = {};
};

// General permutation of a strided input into a contiguous output. `axes[i]`
// names the input dimension that becomes output dimension i.
struct Transpose {
    int64_t numel = 0;
    int32_t ndim  = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]    = {};
    int64_t in_strides[kMaxDims]  = {};
    int64_t out_strides[kMaxDims] = {};
    int64_t axes[kMaxDims]        = {};
};

// ---------------------------------------------------------------------------
// Error handling.
//
// Every launch returns the pending launch error (cudaGetLastError) instead of
// synchronizing; the driver throws with the operation name so the message
// points at the right stack frame. The internal "invalid argument" results use
// the matching cudaError_t values, so a single check covers both.
// ---------------------------------------------------------------------------

inline void check_cuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::format("CUDA error in {}: {} ({})", what,
                                         cudaGetErrorString(err),
                                         cudaGetErrorName(err)));
}

// ---------------------------------------------------------------------------
// Launch entry points. Host drivers and kernels are ordinary functions in the
// same artifact; a launch does not synchronize.
// ---------------------------------------------------------------------------

cudaError_t launch_fill(cudaStream_t stream, void* out, int64_t numel,
                        DataType dtype, double value);
cudaError_t launch_random(cudaStream_t stream, void* out, int64_t numel,
                          DataType dtype, float lo, float hi, uint64_t seed);

// In-place scalar arithmetic: buf = buf <op> scalar.
cudaError_t launch_scalar(cudaStream_t stream, void* buf, int64_t numel,
                          DataType dtype, int op, double scalar);

// Broadcast binary arithmetic into `out`.
cudaError_t launch_binary(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype_a, DataType dtype_b,
                          DataType dtype_out, int op, const Bcast& p);

// Unary element-wise on `numel` contiguous elements. dtype_in and dtype_out
// differ only for the integer -> FLOAT32 promotion the other backends apply to
// transcendental operators; the compute precision follows dtype_in.
cudaError_t launch_unary(cudaStream_t stream, const void* in, void* out,
                         int64_t numel, DataType in_dtype, DataType out_dtype,
                         int op, double p0, double p1);

// Broadcast comparison into an INT8 output. Equal/NotEqual compare with the
// tolerances the host derived from the *original* operand dtypes; ordering
// comparisons are exact and ignore them.
cudaError_t launch_compare(cudaStream_t stream, const void* a, const void* b,
                           void* out, DataType dtype_a, DataType dtype_b, int op,
                           double tol_abs, double tol_rel, const Bcast& p);

// Comparison against a scalar broadcast over a contiguous input.
cudaError_t launch_compare_scalar(cudaStream_t stream, const void* a, void* out,
                                  int64_t numel, DataType dtype, int op,
                                  double scalar, double tol_abs, double tol_rel);

cudaError_t launch_matmul(cudaStream_t stream, const void* a, const void* b,
                          void* out, DataType dtype, int64_t batch, int64_t rows,
                          int64_t cols, int64_t k);

cudaError_t launch_softmax(cudaStream_t stream, const void* in, void* out,
                           DataType in_dtype, DataType out_dtype,
                           const ReduceAxis& p);

// Global reduction: `out` holds one element of out_dtype.
cudaError_t launch_reduce_all(cudaStream_t stream, const void* in, void* out,
                              int64_t numel, DataType in_dtype, DataType out_dtype,
                              int op);

// Axis reduction: `out` is outer * inner elements of out_dtype.
cudaError_t launch_reduce_axis(cudaStream_t stream, const void* in, void* out,
                               DataType in_dtype, DataType out_dtype, int op,
                               const ReduceAxis& p);

// Argmax/argmin along an axis; `out` is INT32 with outer * inner elements.
cudaError_t launch_arg_reduce(cudaStream_t stream, const void* in, void* out,
                              DataType dtype, bool is_max, const ReduceAxis& p);

// All/Any: `out` is one INT32 device element; the host copies it back.
cudaError_t launch_all_any(cudaStream_t stream, const void* in, void* out,
                           int64_t numel, DataType dtype, bool is_all, double value);

// Number of non-zero elements; the result is one INT64 device element.
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

// Materialize a strided view (described by `meta`) into a contiguous buffer.
cudaError_t launch_repack(cudaStream_t stream, const void* in, void* out,
                          const Metadata& meta);

}  // namespace tvv::cuda

#endif  // TVV_CUDA_LAUNCH_HPP
