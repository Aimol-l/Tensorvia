#pragma once

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
inline constexpr int kUnaryErf     = 13;

inline constexpr int kReduceSum  = 0;
inline constexpr int kReduceMin  = 1;
inline constexpr int kReduceMax  = 2;
inline constexpr int kReduceMean = 3;
inline constexpr int kReduceProd = 4;

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
//
// `numel` 是输入缓冲区的真实元素数，轴循环按它截断（limit = min(axis,
// numel - base)）。正常布局下 base + axis*inner <= numel 恒成立，截断因此是
// 恒等的；只有两趟全归约的趟 1（末段短于 axis）会真正用到它。取 0 表示按几何
// outer*axis*inner 推导——直接信一个可能是 0 的字段会让 limit 变负、循环空转、
// 静默算错，所以必须留这条兜底。
//
// `mean_n` 是 kReduceMean 的分母；0 表示用 axis。两趟全归约的趟 2 要把 B 个
// 部分和平均成全归约的元素数 N，而不是趟 2 自己的 axis（=B）。
//
// 该结构经 extern "C" 接缝以指针传给设备单元；两侧同一份定义（设备单元的
// common.h include 本文件），加字段是安全的。
struct ReduceAxis {
    int64_t outer  = 0;
    int64_t axis   = 0;
    int64_t inner  = 0;
    int64_t numel  = 0;
    int64_t mean_n = 0;
};

// sort / topk 的几何：输入视为 outer x n x inner，每个 (outer, inner) 列独立
// 排序，输出排好序的前 keep 个（sort 的 keep = n，topk 的 keep = k）。
// `padded` 是 n 补齐到的 2 的幂（block 路径的 bitonic 网络用）。
struct SortAxis {
    int64_t outer      = 0;
    int64_t n          = 0;
    int64_t inner      = 0;
    int64_t keep       = 0;
    int64_t padded     = 0;
    int32_t descending = 0;  // 1 = 降序（topk 的 largest）
    int32_t reserved   = 0;
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

// 三路广播选择（where）：cond / a / b 各一套广播步长，各自叠加视图偏移。
// `cond_dtype` 是 DataType 的序号；cond 的标量类型在设备侧查表读取。
struct Select {
    int64_t numel      = 0;
    int32_t ndim       = 0;
    int32_t cond_dtype = 0;
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

// conv2d：输入与输出都是连续 NCHW；bias 可为空指针（表示没有偏置）。
struct Conv2d {
    int64_t batch   = 0;
    int64_t in_ch   = 0;
    int64_t in_h    = 0;
    int64_t in_w    = 0;
    int64_t out_ch  = 0;
    int64_t kh      = 0;
    int64_t kw      = 0;
    int64_t out_h   = 0;
    int64_t out_w   = 0;
    int32_t stride  = 1;
    int32_t padding = 0;
};

// pool2d：输入与输出都是连续 NCHW。公共层已把 stride 参数换算成每维的最终
// 步长（stride 缺省取 kernel 尺寸），这里只见到 stride_h/stride_w。
struct Pool2d {
    int64_t batch    = 0;
    int64_t channels = 0;
    int64_t in_h     = 0;
    int64_t in_w     = 0;
    int64_t kh       = 0;
    int64_t kw       = 0;
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t padding  = 0;
    int64_t out_h    = 0;
    int64_t out_w    = 0;
};

// gather：输出形状与 index 相同，非 dim 维坐标决定输入 base。
struct Gather {
    int64_t numel       = 0;
    int32_t ndim        = 0;
    int32_t idx_dtype   = 0;
    int64_t out_shape[kMaxDims]  = {};
    int64_t in_strides[kMaxDims] = {};
    int64_t axis_dim    = 0;
    int64_t axis_stride = 0;
    int32_t dim         = 0;
    int32_t reserved    = 0;
};

static_assert(std::is_trivially_copyable_v<Bcast>, "seam POD");
static_assert(std::is_trivially_copyable_v<Select>, "seam POD");
static_assert(std::is_trivially_copyable_v<IndexAxis>, "seam POD");
static_assert(std::is_trivially_copyable_v<Gather>, "seam POD");
static_assert(std::is_trivially_copyable_v<Conv2d>, "seam POD");
static_assert(std::is_trivially_copyable_v<Pool2d>, "seam POD");
static_assert(std::is_trivially_copyable_v<ReduceAxis>, "seam POD");
static_assert(std::is_trivially_copyable_v<SortAxis>, "seam POD");
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

// 在设备 `device_index` 上创建上下文（含其后所有线程共享的 sycl::context）：
// -1 表示默认设备（第一块 GPU，没有 GPU 时才用 sycl::default_selector_v），
// 0.. 表示 sycl::device::get_devices() 中的对应位置。没有可用设备时返回 nullptr。
// 上下文本身不持有队列：队列按（线程, 上下文）惰性创建，见 sycl_c_current_queue。
void* sycl_c_context_create(int device_index);
void  sycl_c_context_destroy(void* ctx);
// 本进程可见的设备数量。设备索引空间是这个后端自己的：即
// sycl::device::get_devices() 中的位置。SYCL 运行时无法枚举时返回 0。
int   sycl_c_device_count();
// 默认设备的索引（第一块 GPU，没有 GPU 时退回 sycl::default_selector_v），或 -1。
int   sycl_c_default_device_index();
// 最近一次失败在调用线程上的原始描述（sycl 运行时抛出的 what()）。状态码只有
// "失败"三个字，这一句才是能查的东西；没有失败时为 nullptr。
const char* sycl_c_last_error();
// 记录/清空上述描述；启动函数在 catch 里调用，宿主侧不直接调。
void  sycl_c_record_error(const char* what);
// 等待本上下文上**所有线程的队列**清空（设备级同步，对应 torch.cuda.synchronize）。
int   sycl_c_wait(void* ctx);
// 把设备名拷入 `out`；返回写入的字节数。
int   sycl_c_device_name(void* ctx, char* out, int cap);

// 调用线程在这个上下文上的队列句柄；首次调用时惰性创建。返回 nullptr 表示创建
// 失败。句柄归 context 所有，生命周期与 context 相同。
void* sycl_c_current_queue(void* ctx);

// 分配 / 释放 USM 设备内存。`queue` 是 sycl_c_current_queue 返回的句柄；为
// nullptr 时退回调用线程当前的队列。释放会排在 `queue` 上既有工作之后完成，
// 因此调用方传的应当是**分配该块的那条队列**。
void* sycl_c_malloc(void* ctx, void* queue, size_t bytes);
void  sycl_c_free(void* ctx, void* queue, void* ptr);
// 把 dst/src 拷入 `queue` 上并等待其完成：返回即表示数据可被宿主读取。
int   sycl_c_memcpy(void* ctx, void* dst, const void* src, size_t bytes, int kind);

// ---------------------------------------------------------------------------
// 启动函数。每个都在**调用线程的** in-order 队列上提交工作后立即返回；宿主
// 驱动随后调用 sycl_c_wait，与其它后端的同步语义保持一致。
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

// layer_norm / rms_norm：输入视为 outer x n 的连续行，weight（以及 LN 的 bias）
// 是 n 个元素、按行广播；`bias` 为 nullptr 表示没有偏置，`rms` 非 0 时连均值项
// 一并跳过。一次读、一次写完成整行，取代组合式的 8~10 个内核。
int sycl_c_launch_norm(void* ctx, const void* in, const void* weight, const void* bias,
                       void* out, int in_dtype, int out_dtype, int64_t outer, int64_t n,
                       float eps, int rms);

int sycl_c_launch_reduce_axis(void* ctx, const void* in, void* out,
                              int in_dtype, int out_dtype, int op,
                              const tvv::sycl::ReduceAxis* p);
// 全归约趟 1：把 [0, numel) 切成 `blocks` 段，每段一个 work-group 合并读取并在
// local memory 里折成一个部分值，写进 `out`（`blocks` 个 out_dtype 元素）。
int sycl_c_launch_reduce_partial(void* ctx, const void* in, void* out, int64_t numel,
                                 int in_dtype, int out_dtype, int op, int64_t blocks);
int sycl_c_launch_arg_reduce(void* ctx, const void* in, void* out, int dtype,
                             int is_max, const tvv::sycl::ReduceAxis* p);
// sort / topk：每列独立排序后取前 keep 个。`out_vals` 与 `out_idx` 都是
// outer * keep * inner 个元素（vals 为 dtype，idx 为 INT64）。
int sycl_c_launch_sort(void* ctx, const void* in, void* out_vals, void* out_idx,
                       int dtype, const tvv::sycl::SortAxis* p);
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

// where：cond 判真后从 a 或 b 取元素；写出 dtype 由 out_dtype 决定（BOOL 规范化）。
int sycl_c_launch_select(void* ctx, const void* cond, const void* a, const void* b,
                         void* out, int out_dtype, const tvv::sycl::Select* p);

// index_select / gather：`flag` 非空时，任一索引落在 [0, axis_dim) 之外置 1。
// 设备侧对越界索引做截断，因此即使调用方不读标志也不会越界读写。
int sycl_c_launch_index_select(void* ctx, const void* in, const void* idx, void* out,
                               int out_dtype, int idx_dtype,
                               const tvv::sycl::IndexAxis* p, int32_t* flag);
int sycl_c_launch_gather(void* ctx, const void* in, const void* idx, void* out,
                         int out_dtype, int idx_dtype, const tvv::sycl::Gather* p,
                         int32_t* flag);

// conv2d：输入/输出都是连续 NCHW；`bias` 可为 nullptr。
int sycl_c_launch_conv2d(void* ctx, const void* in, const void* weight, const void* bias,
                         void* out, int dtype, const tvv::sycl::Conv2d* p);
// pool2d：average 非 0 为 AvgPool，否则为 MaxPool。
int sycl_c_launch_pool2d(void* ctx, const void* in, void* out, int dtype,
                         const tvv::sycl::Pool2d* p, int average);

// 沿 axis 的累积和；输出与输入同形同 dtype。
int sycl_c_launch_cumsum(void* ctx, const void* in, void* out, int dtype,
                         const tvv::sycl::IndexAxis* p);

}  // extern "C"