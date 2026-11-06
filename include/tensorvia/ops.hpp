#pragma once

#include <cstddef>
#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"

namespace ops {

template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

void println(Tensor& a);
void println(Tensor&& a);

void synchronize();
void empty_cache();

int device_count(Device kind);
int current_device(Device kind);
void set_device(Device kind, int index);

class DeviceGuard {
private:
    DeviceId m_previous;
public:
    explicit DeviceGuard(DeviceId id);
    DeviceGuard(Device kind, int index) : DeviceGuard(DeviceId{kind, index}) {}
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

void export_csv(Tensor& a, const std::string& path);
void export_csv(Tensor&& a, const std::string& path);

Tensor Ones(const std::vector<int64_t>& shape, DataType dtype, Device device = Device::CPU);
Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, Device device = Device::CPU);
Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float value, Device device = Device::CPU);
Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, Device device = Device::CPU);
Tensor Randn(const std::vector<int64_t>& shape, DataType dtype, float mean = 0.0f, float std = 1.0f,
             Device device = Device::CPU);

// TensorOptions 形式（FR-7）：dtype 与设备实例一次传入。
// 未指定设备索引时，取调用线程在该后端上的 current device。
Tensor Ones(const std::vector<int64_t>& shape, const TensorOptions& options);
Tensor Zeros(const std::vector<int64_t>& shape, const TensorOptions& options);
Tensor Fill(const std::vector<int64_t>& shape, float value, const TensorOptions& options);
Tensor Random(const std::vector<int64_t>& shape, float min, float max, const TensorOptions& options);
Tensor Randn(const std::vector<int64_t>& shape, float mean, float std, const TensorOptions& options);
// DeviceId 形式：直接点名设备实例，不做线程私有查找。
Tensor Ones(const std::vector<int64_t>& shape, DataType dtype, DeviceId device);
Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, DeviceId device);
Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float value, DeviceId device);
Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, DeviceId device);

Tensor Arange(float start, float stop, float step = 1.0f, DataType dtype = default_dtype());
Tensor Linspace(float start, float stop, int64_t steps, DataType dtype = default_dtype());
Tensor Eye(int64_t n, DataType dtype = default_dtype());
Tensor Eye(int64_t n, int64_t m, DataType dtype = default_dtype());
// 生成器的设备形式：值在宿主生成后一次性搬到目标设备，因此同一调用在各后端
// 取值完全一致（randn 也一样——RNG 只在宿主保留一份实现）。
Tensor Arange(float start, float stop, float step, const TensorOptions& options);
Tensor Linspace(float start, float stop, int64_t steps, const TensorOptions& options);
Tensor Eye(int64_t n, const TensorOptions& options);
Tensor Eye(int64_t n, int64_t m, const TensorOptions& options);

void Zeros(Tensor& a);
void Ones(Tensor& a);
void Fill(Tensor& a, float value);

Tensor Slice(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges);

void Add(Tensor& a, float b);
void Sub(Tensor& a, float b);
void Mul(Tensor& a, float b);
void Div(Tensor& a, float b);

Tensor Add(const Tensor& a, float b);
Tensor Sub(const Tensor& a, float b);
Tensor Mul(const Tensor& a, float b);
Tensor Div(const Tensor& a, float b);

Tensor Add(const Tensor& a, const Tensor& b);
Tensor Sub(const Tensor& a, const Tensor& b);
Tensor Mul(const Tensor& a, const Tensor& b);
Tensor Div(const Tensor& a, const Tensor& b);

void Add(const Tensor& a, const Tensor& b, Tensor& dst);
void Sub(const Tensor& a, const Tensor& b, Tensor& dst);
void Mul(const Tensor& a, const Tensor& b, Tensor& dst);
void Div(const Tensor& a, const Tensor& b, Tensor& dst);

// N-D 批处理矩阵乘，语义与 PyTorch 一致：最后两维是矩阵本身，1-D 操作数按行/列
// 向量处理，前导维右对齐广播。
Tensor MatMul(const Tensor& a, const Tensor& b);
// 批处理矩阵乘，要求两个操作数都是 3-D 且 batch 维一致。
Tensor Bmm(const Tensor& a, const Tensor& b);
// beta * input + alpha * (mat1 @ mat2)；`input` 可广播到结果形状。
Tensor Addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
             float beta = 1.0f, float alpha = 1.0f);
// y = x @ weight^T (+ bias) —— 即 nn.Linear，支持任意多个前导 batch 维。
Tensor Linear(const Tensor& input, const Tensor& weight);
Tensor Linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

// 2-D 卷积（cross-correlation，与 torch.nn.functional.conv2d 一致）。
// input 为 [N, C, H, W]，也接受不带 batch 的 [C, H, W]（此时输出不带 batch 维）；
// weight 为 [Cout, C, kh, kw]，bias 可选、形状 [Cout]。stride 与 padding 同时
// 作用于 H/W 两维，不做 dilation 与 groups：
//   Hout = (H + 2 * padding - kh) / stride + 1
// 四个浮点 dtype；输入、权重、bias 必须同 dtype 同设备。
Tensor Conv2d(const Tensor& input, const Tensor& weight, int64_t stride = 1,
              int64_t padding = 0);
Tensor Conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int64_t stride = 1, int64_t padding = 0);

// 2-D 池化。input 为 [N, C, H, W]，也接受不带 batch 的 [C, H, W]（此时输出不
// 带 batch 维）；四个浮点 dtype，半精度在 FLOAT32 里累加（Avg）后写回。
//   Hout = (H + 2 * padding - kh) / stride + 1，Wout 同理
// `stride` 为 0 表示取 kernel 尺寸（PyTorch 的 stride=None 心智：2x2 池化默认
// 步长 2）；为正时同时作用于 H/W。padding 区域不参与取值：Max 忽略，Avg 计 0
// 且计入分母（与 PyTorch count_include_pad=True 一致）。窗口整体越界时输出 0。
Tensor MaxPool2d(const Tensor& input, int64_t kh, int64_t kw, int64_t stride = 0,
                 int64_t padding = 0);
Tensor AvgPool2d(const Tensor& input, int64_t kh, int64_t kw, int64_t stride = 0,
                 int64_t padding = 0);

// 融合块：MaxPool2d(Relu(Conv2d(x, w, b, stride, padding)), 2, 2) 的精确等价。
// [N, C, H, W]（或 [C, H, W]）+ weight [Cout, C, kh, kw]（+ bias [Cout]） ->
// [N, Cout, H', W']，pool 固定 2x2/stride 2：H' = (Hout - 2) / 2 + 1。
// 与组合式逐位一致（测试锁定）；存在的理由是省掉两个算子的提交与同步往返。
Tensor Conv2dReluMaxPool2d(const Tensor& input, const Tensor& weight,
                           int64_t stride = 1, int64_t padding = 0);
Tensor Conv2dReluMaxPool2d(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, int64_t stride = 1, int64_t padding = 0);

void Abs(Tensor& a);
void Relu(Tensor& a);
void Sin(Tensor& a);
void Cos(Tensor& a);
void Tan(Tensor& a);
void Silu(Tensor& a);
void Tanh(Tensor& a);
void Sigmoid(Tensor& a);
void Pow(Tensor& a, float val);
void Log(Tensor& a, float val);
void Clamp(Tensor& a, float min, float max);

Tensor Abs(const Tensor& a);
Tensor Sin(const Tensor& a);
Tensor Cos(const Tensor& a);
Tensor Tan(const Tensor& a);
Tensor Exp(const Tensor& a);
Tensor Relu(const Tensor& a);
Tensor Silu(const Tensor& a);
Tensor Tanh(const Tensor& a);
Tensor Sqrt(const Tensor& a);
Tensor Sigmoid(const Tensor& a);
Tensor Pow(const Tensor& a, float val);
Tensor Log(const Tensor& a, float val);
Tensor Softmax(const Tensor& a, int axis);
Tensor LogSoftmax(const Tensor& a, int axis);
Tensor Erf(const Tensor& a);
Tensor Clamp(const Tensor& a, float min, float max);

Tensor LayerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps = 1e-5f);
Tensor RmsNorm(const Tensor& a, const Tensor& weight, int axis, float eps = 1e-5f);

// 归约返回张量而不是宿主标量。全归约得到 0-dim（rank-0）张量——与逐轴归约的
// 形状规则一致——值留在原设备上，直到 item<T>() 显式取回。结果 dtype：
// Sum/Prod 走 reduction_type，Mean/Var/Std/Norm 走 float_reduction_type，
// Min/Max 保持输入 dtype。负轴从末尾计数。
Tensor Sum(const Tensor& a);
Tensor Min(const Tensor& a);
Tensor Max(const Tensor& a);
Tensor Mean(const Tensor& a);
Tensor Prod(const Tensor& a);
Tensor Var(const Tensor& a);
Tensor Std(const Tensor& a);
Tensor Norm(const Tensor& a, float p = 2.0f);

Tensor Sum(const Tensor& a, int axis, bool keepdim = false);
Tensor Min(const Tensor& a, int axis, bool keepdim = false);
Tensor Max(const Tensor& a, int axis, bool keepdim = false);
Tensor Mean(const Tensor& a, int axis, bool keepdim = false);
Tensor Prod(const Tensor& a, int axis, bool keepdim = false);
Tensor Var(const Tensor& a, int axis, bool keepdim = false);
Tensor Std(const Tensor& a, int axis, bool keepdim = false);
Tensor Norm(const Tensor& a, float p, int axis, bool keepdim = false);
Tensor Cumsum(const Tensor& a, int axis);

// 多轴归约（FR-3）：所列轴一起被移除，负轴从末尾计数，空列表等价于“所有轴”。
// 逐元素归约按一次折叠一轴实现；Var/Std/Norm 不能这样折叠，多轴形式显式报错。
Tensor Sum(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Sum(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim = false);
Tensor Min(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Min(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim = false);
Tensor Max(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Max(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim = false);
Tensor Mean(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Mean(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim = false);
Tensor Prod(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Prod(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim = false);
Tensor Var(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Std(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim = false);
Tensor Norm(const Tensor& a, float p, std::initializer_list<int64_t> dims, bool keepdim = false);

// 带 dtype 限定的归约：先按归约自身的 dtype 规则求值，再把结果 cast 到 `dtype`。
// 所有归约都拒绝 BOOL 作为输出 dtype：归约结果是数值，谓词形式请用 All/Any。
Tensor Sum(const Tensor& a, DataType dtype);
Tensor Sum(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Sum(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype);
Tensor Min(const Tensor& a, DataType dtype);
Tensor Min(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Min(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype);
Tensor Max(const Tensor& a, DataType dtype);
Tensor Max(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Max(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype);
Tensor Mean(const Tensor& a, DataType dtype);
Tensor Mean(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Mean(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype);
Tensor Prod(const Tensor& a, DataType dtype);
Tensor Prod(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Prod(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype);
Tensor Var(const Tensor& a, DataType dtype);
Tensor Var(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Std(const Tensor& a, DataType dtype);
Tensor Std(const Tensor& a, int axis, bool keepdim, DataType dtype);
Tensor Norm(const Tensor& a, float p, DataType dtype);
Tensor Norm(const Tensor& a, float p, int axis, bool keepdim, DataType dtype);

// 损失：返回 0 维张量（与归约一致），dtype 走 mean 的规则——整型与 16 位浮点
// 输入回到 FLOAT32，FLOAT64 保持 FLOAT64。
Tensor MseLoss(const Tensor& input, const Tensor& target);
Tensor NllLoss(const Tensor& input, const Tensor& target);
Tensor CrossEntropyLoss(const Tensor& input, const Tensor& target);

void Typecast(Tensor& a, DataType dst_type);
Tensor Typecast(const Tensor& a, DataType dst_type);
Tensor Concat(const std::vector<Tensor>& tensors, int dim);

void Transpose(Tensor& a);
Tensor Transpose(const Tensor& a, std::initializer_list<int64_t> axes);
void Transpose(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes);

Tensor Equal(const Tensor& a, const Tensor& b);
Tensor NotEqual(const Tensor& a, const Tensor& b);
Tensor Greater(const Tensor& a, const Tensor& b);
Tensor Less(const Tensor& a, const Tensor& b);
Tensor GreaterEqual(const Tensor& a, const Tensor& b);
Tensor LessEqual(const Tensor& a, const Tensor& b);

Tensor Equal(const Tensor& a, float val);
Tensor NotEqual(const Tensor& a, float val);
Tensor Greater(const Tensor& a, float val);
Tensor Less(const Tensor& a, float val);
Tensor GreaterEqual(const Tensor& a, float val);
Tensor LessEqual(const Tensor& a, float val);

Tensor Where(const Tensor& cond, const Tensor& a, const Tensor& b);
Tensor Where(const Tensor& cond, const Tensor& a, float b);
Tensor Where(const Tensor& cond, float a, const Tensor& b);

// all/any：对全部元素做真值判定（“非零即真”，与 Python 一致），返回输入设备上的
// 0-dim BOOL 张量；空张量按空真/空假处理。下面带两个参数的旧形式是“每个元素都
// 等于 val”的谓词，返回宿主 bool。
Tensor All(const Tensor& t);
Tensor Any(const Tensor& t);
bool All(const Tensor& t, float val);
bool Any(const Tensor& t, float val);

// 非零元素的下标，行主序：形状 (count, dims)，INT64（与 PyTorch 的 nonzero 一致）。
// 空张量得到 (0, dims)。只要计数时用 CountNonzero——那是旧版 Nonzero 的语义。
Tensor Nonzero(const Tensor& t);
size_t CountNonzero(const Tensor& t);

// BOOL 掩码选择与填充（FR-5）。mask 必须能广播到输入形状，可为 BOOL 或任意整型
// （非零即选中；INT8 是旧版谓词类型）。MaskedSelect 按行主序返回选中元素的 1-D
// 张量。两者目前都只有 CPU 实现，设备上调用会明确报错。
Tensor MaskedSelect(const Tensor& input, const Tensor& mask);
Tensor MaskedFill(const Tensor& input, const Tensor& mask, float value);
Tensor Argmax(const Tensor& t, int axis, bool keepdim = false);
Tensor Argmin(const Tensor& t, int axis, bool keepdim = false);
Tensor IndexSelect(const Tensor& input, int dim, const Tensor& index);
Tensor Gather(const Tensor& input, int dim, const Tensor& index);

// ---------------------------------------------------------------------------
// 形状 / 数据搬运算子（FR-11）。
// ---------------------------------------------------------------------------
// Split/Chunk 是纯视图：共享 storage、同一设备，不跑内核。Chunk 的块长为
// ceil(size / chunks)，返回的视图数可能少于 `chunks`（与 PyTorch 一致）。
std::vector<Tensor> Split(const Tensor& t, int64_t split_size, int64_t dim = 0);
std::vector<Tensor> Chunk(const Tensor& t, int64_t chunks, int64_t dim = 0);
// 沿新维度堆叠；所有输入的形状、dtype 与设备必须一致。
Tensor Stack(const std::vector<Tensor>& tensors, int64_t dim = 0);
// 常量填充版 torch.nn.functional.pad。`pads` 是从最后一维开始的扁平
// (begin, end) 对列表；负数对表示裁剪。
Tensor Pad(const Tensor& t, const std::vector<int64_t>& pads, float value = 0.0f);
// NumPy `tile`：第 i 维重复 repeats[i] 次。`repeats` 可以比操作数的 rank 长
// （多出的前导维按长度 1 处理）。
Tensor Repeat(const Tensor& t, const std::vector<int64_t>& repeats);
Tensor Flip(const Tensor& t, const std::vector<int64_t>& dims);
Tensor Roll(const Tensor& t, int64_t shift, int64_t dim);

// Sort 与 topk 返回排序后的值以及产生这些值的下标（INT64）。两者目前都只有 CPU
// 实现；设备上调用走 FR-8 的“后端未实现”错误路径。
struct SortResult {
    Tensor values;
    Tensor indices;
};
struct TopkResult {
    Tensor values;
    Tensor indices;
};
SortResult Sort(const Tensor& t, int64_t dim = -1, bool descending = false);
TopkResult Topk(const Tensor& t, int64_t k, int64_t dim = -1, bool largest = true);

}  // namespace ops
