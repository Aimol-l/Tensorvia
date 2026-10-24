// The seam between the host drivers and the SYCL device units.
//
// SYCL in this project is a two-compiler model: the host translation units are
// built by the project's C++ toolchain (modules under mcpp, generated headers
// under CMake), while everything that touches a queue or a kernel lives in
// `device/*.sycl` and is built by the DPC++ compiler. The two runtimes do not
// share C++ entities, so this header is the whole contract between them:
//
//   * fixed-width integers, plain pointers and POD structs only;
//   * no std containers, no std::string, no tensorvia types;
//   * `extern "C"` entry points; a failure is a negative status code, never an
//     exception crossing the boundary.
//
// Both build paths use this exact header, so the CMake and mcpp artifacts run
// the same code with the same seam.
#ifndef TVV_SYCL_ISLAND_H
#define TVV_SYCL_ISLAND_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tvv::sycl {

inline constexpr int32_t kMaxDims = 8;

// Scalar tags. The values match tensorvia's DataType enumerators, so the host
// can pass `static_cast<int>(dtype)` without a lookup table; the device side
// uses these names. Keeping the mapping documented here is the point.
enum DType : int32_t {
    kInt8 = 0,
    kInt16,
    kInt32,
    kInt64,
    kBfloat16,
    kFloat16,
    kFloat32,
    kFloat64,
};

// ---------------------------------------------------------------------------
// Operator codes shared by the host drivers and the kernel switches.
// ---------------------------------------------------------------------------

inline constexpr int kBinAdd = 0;
inline constexpr int kBinSub = 1;
inline constexpr int kBinMul = 2;
inline constexpr int kBinDiv = 3;

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

inline constexpr int kReduceSum  = 0;
inline constexpr int kReduceMin  = 1;
inline constexpr int kReduceMax  = 2;
inline constexpr int kReduceMean = 3;

inline constexpr int kCmpEqual        = 0;
inline constexpr int kCmpNotEqual     = 1;
inline constexpr int kCmpGreater      = 2;
inline constexpr int kCmpLess         = 3;
inline constexpr int kCmpGreaterEqual = 4;
inline constexpr int kCmpLessEqual    = 5;

// Copy directions for sycl_c_memcpy.
inline constexpr int kHostToDevice = 0;
inline constexpr int kDeviceToHost = 1;
inline constexpr int kDeviceToDevice = 2;

// ---------------------------------------------------------------------------
// Parameter structs. Everything is passed by pointer to a launch entry point
// and captured by value into the kernel; no managed-memory staging is
// involved.
// ---------------------------------------------------------------------------

// Broadcast descriptor for binary element-wise and comparison kernels.
struct Bcast {
    int64_t numel    = 0;
    int32_t ndim     = 0;
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
    int64_t numel      = 0;
    int64_t offset     = 0;  // view origin, in elements
    int32_t ndim       = 0;
    int32_t dtype_size = 0;
    int64_t shape[kMaxDims]   = {};
    int64_t strides[kMaxDims] = {};
};

// Slice of a strided input.
struct Slice {
    int64_t numel    = 0;
    int64_t offset   = 0;  // input view origin, in elements
    int32_t ndim     = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]   = {};
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t starts[kMaxDims]     = {};
};

// General permutation of a strided input into a contiguous output. `axes[i]`
// names the input dimension that becomes output dimension i.
struct Transpose {
    int64_t numel    = 0;
    int32_t ndim     = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]    = {};
    int64_t in_strides[kMaxDims]  = {};
    int64_t out_strides[kMaxDims] = {};
    int64_t axes[kMaxDims]        = {};
};

static_assert(std::is_trivially_copyable_v<Bcast>, "seam POD");
static_assert(std::is_trivially_copyable_v<ReduceAxis>, "seam POD");
static_assert(std::is_trivially_copyable_v<Repack>, "seam POD");
static_assert(std::is_trivially_copyable_v<Slice>, "seam POD");
static_assert(std::is_trivially_copyable_v<Transpose>, "seam POD");

// Status codes. 0 is success; every other value is an error the host turns
// into an exception carrying the operation name.
inline constexpr int kSyclOk          = 0;
inline constexpr int kSyclInvalid     = -1;
inline constexpr int kSyclUnsupported = -2;
inline constexpr int kSyclFailure     = -3;

}  // namespace tvv::sycl

extern "C" {

// ---------------------------------------------------------------------------
// Context, memory and host transfers.
// ---------------------------------------------------------------------------

// Creates a queue with sycl::default_selector_v (GPU preferred, OpenCL CPU as
// the fallback). Returns nullptr when no device is usable.
void* sycl_c_context_create();
void  sycl_c_context_destroy(void* ctx);
int   sycl_c_wait(void* ctx);
// Copies the device name into `out`; returns the number of bytes written.
int   sycl_c_device_name(void* ctx, char* out, int cap);

void* sycl_c_malloc(void* ctx, size_t bytes);
void  sycl_c_free(void* ctx, void* ptr);
int   sycl_c_memcpy(void* ctx, void* dst, const void* src, size_t bytes, int kind);

// ---------------------------------------------------------------------------
// Launches. Each submits work on the in-order queue and returns immediately;
// the host driver calls sycl_c_wait afterwards, matching the other backends'
// synchronous semantics.
// ---------------------------------------------------------------------------

int sycl_c_launch_fill(void* ctx, void* out, int64_t numel, int dtype, double value);
int sycl_c_launch_random(void* ctx, void* out, int64_t numel, int dtype,
                         float lo, float hi, uint64_t seed);

int sycl_c_launch_scalar(void* ctx, void* buf, int64_t numel, int dtype,
                         int op, double scalar);
int sycl_c_launch_binary(void* ctx, const void* a, const void* b, void* out,
                         int dtype, int op, const tvv::sycl::Bcast* p);
int sycl_c_launch_unary(void* ctx, const void* in, void* out, int64_t numel,
                        int in_dtype, int out_dtype, int op, double p0, double p1);
int sycl_c_launch_compare(void* ctx, const void* a, const void* b, void* out,
                          int dtype, int op, double tol_abs, double tol_rel,
                          const tvv::sycl::Bcast* p);
int sycl_c_launch_compare_scalar(void* ctx, const void* a, void* out, int64_t numel,
                                 int dtype, int op, double scalar,
                                 double tol_abs, double tol_rel);

int sycl_c_launch_matmul(void* ctx, const void* a, const void* b, void* out,
                         int dtype, int64_t batch, int64_t rows, int64_t cols,
                         int64_t k);
int sycl_c_launch_softmax(void* ctx, const void* in, void* out,
                          int in_dtype, int out_dtype,
                          const tvv::sycl::ReduceAxis* p);

int sycl_c_launch_reduce_all(void* ctx, const void* in, void* out, int64_t numel,
                             int in_dtype, int out_dtype, int op);
int sycl_c_launch_reduce_axis(void* ctx, const void* in, void* out,
                              int in_dtype, int out_dtype, int op,
                              const tvv::sycl::ReduceAxis* p);
int sycl_c_launch_arg_reduce(void* ctx, const void* in, void* out, int dtype,
                             int is_max, const tvv::sycl::ReduceAxis* p);
int sycl_c_launch_all_any(void* ctx, const void* in, void* out, int64_t numel,
                          int dtype, int is_all, double value);
// `out` is one INT32 element holding the count.
int sycl_c_launch_nonzero(void* ctx, const void* in, void* out, int64_t numel,
                          int dtype);

int sycl_c_launch_cast(void* ctx, const void* in, void* out, int64_t numel,
                       int in_dtype, int out_dtype);

int sycl_c_launch_concat(void* ctx, void* out, const void* const* inputs,
                         const int64_t* input_offsets, int32_t num_inputs,
                         int dtype, const int64_t* out_strides,
                         const int64_t* in_shapes, const int64_t* in_strides,
                         int32_t ndim, int32_t axis);
int sycl_c_launch_slice(void* ctx, const void* in, void* out, int dtype,
                        const tvv::sycl::Slice* p);
int sycl_c_launch_transpose(void* ctx, const void* in, void* out, int dtype,
                            const tvv::sycl::Transpose* p);
int sycl_c_launch_repack(void* ctx, const void* in, void* out, int64_t numel,
                         int64_t offset, int32_t ndim, int32_t dtype_size,
                         const int64_t* shape, const int64_t* strides);

}  // extern "C"

#endif  // TVV_SYCL_ISLAND_H
