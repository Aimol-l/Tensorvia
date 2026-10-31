// 宿主驱动与 SYCL 设备单元之间的接缝。
//
// 本项目里的 SYCL 是双编译器模型：宿主翻译单元由项目自身的 C++ 工具链构建
// （mcpp 下是模块，CMake 下是生成的头文件），而所有触及 queue 或 kernel 的代码
// 都在 `device/*.sycl` 中、由 DPC++ 编译器构建。两个运行时不共享 C++ 实体，所以
// 这个头文件就是它们之间的全部契约：
//
//   * 只用定宽整数、裸指针与 POD 结构体；
//   * 不用 std 容器、std::string、tensorvia 类型；
//   * 入口点是 `extern "C"`；失败以负状态码表示，绝不让异常跨越边界。
//
// 两条构建路径使用同一个头文件，因此 CMake 与 mcpp 产物跑的是同一份代码、同一个
// 接缝。
#ifndef TVV_SYCL_ISLAND_H
#define TVV_SYCL_ISLAND_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tvv::sycl {

inline constexpr int32_t kMaxDims = 8;

// 标量标签。取值与 tensorvia 的 DataType 枚举一致，宿主可直接传
// `static_cast<int>(dtype)` 而无需查表；设备侧使用这些名字。把映射记录在此正是
// 本段的目的。
enum DType : int32_t {
    kBool = 0,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kBfloat16,
    kFloat16,
    kFloat32,
    kFloat64,
};

// ---------------------------------------------------------------------------
// 宿主驱动与内核 switch 共享的算子码。
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
inline constexpr int kUnaryLog     = 6;   // p0 = 对数底
inline constexpr int kUnaryPow     = 7;   // p0 = 指数
inline constexpr int kUnaryRelu    = 8;
inline constexpr int kUnarySilu    = 9;
inline constexpr int kUnaryTanh    = 10;
inline constexpr int kUnarySigmoid = 11;
inline constexpr int kUnaryClamp   = 12;  // p0 = 下界，p1 = 上界

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

// sycl_c_memcpy 的拷贝方向。
inline constexpr int kHostToDevice = 0;
inline constexpr int kDeviceToHost = 1;
inline constexpr int kDeviceToDevice = 2;

// ---------------------------------------------------------------------------
// 参数结构体。所有内容以指针传给启动入口，再按值捕获进内核；不涉及托管内存暂存。
// ---------------------------------------------------------------------------

// 二元逐元素与比较内核的广播描述符。
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

// 单轴归约的几何形状：输入视为 outer x axis x inner。
struct ReduceAxis {
    int64_t outer = 0;
    int64_t axis  = 0;
    int64_t inner = 0;
};

// 跨步视图重排（clone_as_contiguous）。
struct Repack {
    int64_t numel      = 0;
    int64_t offset     = 0;  // 视图起点，以元素计
    int32_t ndim       = 0;
    int32_t dtype_size = 0;
    int64_t shape[kMaxDims]   = {};
    int64_t strides[kMaxDims] = {};
};

// 跨步输入的切片。
struct Slice {
    int64_t numel    = 0;
    int64_t offset   = 0;  // 输入视图起点，以元素计
    int32_t ndim     = 0;
    int32_t reserved = 0;
    int64_t in_shape[kMaxDims]   = {};
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t starts[kMaxDims]     = {};
};

// 把跨步输入按任意置换重排为连续输出。`axes[i]` 是成为输出第 i 维的输入维。
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

// 状态码。0 表示成功；其它值都是错误，宿主会把它转成带算子名的异常。
inline constexpr int kSyclOk          = 0;
inline constexpr int kSyclInvalid     = -1;
inline constexpr int kSyclUnsupported = -2;
inline constexpr int kSyclFailure     = -3;

}  // namespace tvv::sycl

extern "C" {

// ---------------------------------------------------------------------------
// 上下文、内存与宿主传输。
// ---------------------------------------------------------------------------

// 在设备 `device_index` 上创建队列：-1 表示 sycl::default_selector_v（优先 GPU，
// 以 OpenCL CPU 兜底），0.. 表示 sycl::device::get_devices() 中的对应位置。没有
// 可用设备时返回 nullptr。
void* sycl_c_context_create(int device_index);
void  sycl_c_context_destroy(void* ctx);
// 本进程可见的设备数量。设备索引空间是这个后端自己的：即
// sycl::device::get_devices() 中的位置。SYCL 运行时无法枚举时返回 0。
int   sycl_c_device_count();
// sycl::default_selector_v 会选中的设备索引，或 -1。
int   sycl_c_default_device_index();
int   sycl_c_wait(void* ctx);
// 把设备名拷入 `out`；返回写入的字节数。
int   sycl_c_device_name(void* ctx, char* out, int cap);

void* sycl_c_malloc(void* ctx, size_t bytes);
void  sycl_c_free(void* ctx, void* ptr);
int   sycl_c_memcpy(void* ctx, void* dst, const void* src, size_t bytes, int kind);

// ---------------------------------------------------------------------------
// 启动函数。每个都在 in-order 队列上提交工作后立即返回；宿主驱动随后调用
// sycl_c_wait，与其它后端的同步语义保持一致。
// ---------------------------------------------------------------------------

int sycl_c_launch_fill(void* ctx, void* out, int64_t numel, int dtype, double value);
int sycl_c_launch_random(void* ctx, void* out, int64_t numel, int dtype,
                         float lo, float hi, uint64_t seed);

int sycl_c_launch_scalar(void* ctx, void* buf, int64_t numel, int dtype,int op, double scalar);

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
// `out` 是一个存放计数的 INT32 元素。
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
