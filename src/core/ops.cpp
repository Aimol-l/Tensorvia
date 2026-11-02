#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/ops.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/backend/cpu/cpu_ops.hpp"
#include "tensorvia/backend/vulkan/vulkan_ops.hpp"
#include "tensorvia/backend/cuda/cuda_ops.hpp"
#include "tensorvia/backend/sycl/sycl_ops.hpp"

namespace {

template <template <Device> class Impl>
concept has_vulkan_impl = requires { sizeof(Impl<Device::VULKAN>); };

template <template <Device> class Impl>
concept has_cuda_impl = requires { sizeof(Impl<Device::CUDA>); };

template <template <Device> class Impl>
concept has_sycl_impl = requires { sizeof(Impl<Device::SYCL>); };

// 后端派发：Impl<Device::X> 的全特化存在就走它，否则抛"该后端未实现"。
// 注意：这条通用抛错**不带算子名**（spec Test Steps 4 要求带算子名）。新增
// CPU-only 算子请用 require_cpu(t, "op_name") 显式拒绝（消息含算子名与后端），
// 不要依赖这里的通用消息；现有的 Sort/Topk 等已按此办理。
template <template <Device> class Impl, typename... Args>
decltype(auto) execute_on(DeviceId device, Args&&... args) {
    ops::DeviceGuard guard(device);
    switch (device.kind) {
    case Device::CPU:
        return Impl<Device::CPU>::execute(std::forward<Args>(args)...);
#if defined(BACKEND_VULKAN)
    case Device::VULKAN:
        if constexpr (has_vulkan_impl<Impl>) {
            return Impl<Device::VULKAN>::execute(std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("this operation is not implemented on the VULKAN backend");
        }
#endif
#if defined(BACKEND_CUDA)
    case Device::CUDA:
        if constexpr (has_cuda_impl<Impl>) {
            return Impl<Device::CUDA>::execute(std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("this operation is not implemented on the CUDA backend");
        }
#endif
#if defined(BACKEND_SYCL)
    case Device::SYCL:
        if constexpr (has_sycl_impl<Impl>) {
            return Impl<Device::SYCL>::execute(std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("this operation is not implemented on the SYCL backend");
        }
#endif
    default:
        throw std::runtime_error(std::format(
            "no implementation compiled for device {}", device_to_string(device.kind)));
    }
}

// 裸 `Device` 只命名后端、不命名实例：创建类算子落在调用线程在该后端上的 current
// 实例，与 torch 的工厂函数一致。
inline DeviceId current_id(Device device) {
    return DeviceId{device, current_device_index(device)};
}

inline void require_cpu(const Tensor& t, std::string_view op) {
    if (t.device() != Device::CPU)
        throw std::runtime_error(std::format(
            "{}: only the CPU backend implements this op (tensor is on {})",
            op, device_to_string(t.device())));
}

// Vulkan 内核绑定的是 impl 的整个 buffer，shader 从 buffer 基址按下标寻址，
// 描述符里没有视图 offset；其余后端的内核直接读 Tensor::data()（已含 offset），
// 连续视图可以零拷贝直通。所以只有 Vulkan 上"有 offset 的视图"需要先物化成
// 自己的 buffer。
bool needs_offset_free_input(const Tensor& t) {
    return t.device() == Device::VULKAN && t.offset() != 0;
}

// slice()/t()/permute() 返回共享底层 buffer 的视图，而计算 kernel 只支持从
// Tensor::data()（已含视图 offset）起线性遍历 numel 个元素：
//  - 连续视图（含带 offset 的切片）：线性遍历正好命中视图元素，零成本直通；
//    Vulkan 除外（见 needs_offset_free_input），必须先物化；
//  - 非连续视图（t()/permute()/跨步切片）：先物化成连续拷贝再进 kernel，
//    否则线性遍历会读到视图之外的共享内存。
// 这是 strided 内核接入前的正确性守卫；storage 保活物化结果直到调用结束。
const Tensor& linearize(const Tensor& t, Tensor& storage) {
    if (t.is_contiguous() && !needs_offset_free_input(t)) return t;
    // clone() 对非连续视图走 clone_as_contiguous，对带 offset 的连续视图走
    // 物化拷贝：两种情况下结果都是 offset 为 0 的独立 buffer。
    storage = t.clone();
    return storage;
}

// 归约输入准备：CPU 上被归约轴在内存里连续（stride 1）时直接吃视图——每个输出
// 元素就是一次连续扫描，先拷一遍整个张量只是白花带宽。其余情况物化：被归约轴跨步
// 的按列读是缓存最坏的形态（一次拷贝更便宜），设备后端的内核也按连续
// (outer, axis, inner) 布局寻址。
const Tensor& prepare_reduce_input(const Tensor& t, Tensor& storage, int axis) {
    if (t.device() == Device::CPU && !t.is_contiguous() && t.strides()[axis] == 1)
        return t;
    return linearize(t, storage);
}

// 写入型（in-place / dst 输出）操作从 Tensor::data() 起线性读写 numel 个元素：
//  - 非连续视图：读写会越过视图落在共享内存的错误元素上，也无法保持"写穿到
//    原张量"的语义 —— 拒绝；
//  - Vulkan 带 offset 的视图：内核只拿到 impl 的 buffer，写入从基址开始，
//    会写坏同一 storage 里别人的元素 —— 拒绝（其余后端内核读 Tensor::data()，
//    offset 视图按视图元素原地更新是正确的）。
// 两端都走明确报错路径，而不是静默写坏数据。
void require_contiguous(const Tensor& t, std::string_view op) {
    if (!t.is_contiguous())
        throw std::runtime_error(std::format(
            "{}: non-contiguous view is not supported here; call to_contiguous() first", op));
    if (t.device() == Device::VULKAN && t.offset() != 0)
        throw std::runtime_error(std::format(
            "{}: in-place update of a view with a storage offset is not supported on Vulkan; "
            "clone() it into a standalone buffer first", op));
}

// BOOL 是谓词 dtype：比较、where、索引、数据搬移、归约与 cast 合法，数值算术不合法。
// 在公开入口处拒绝，让每个后端看到同一份契约。
void require_not_bool(const Tensor& t, std::string_view op) {
    if (is_bool_dtype(t.dtype()))
        throw std::runtime_error(std::format(
            "{}: boolean arithmetic is not supported; cast to a numeric dtype first", op));
}

void require_not_bool(const Tensor& a, const Tensor& b, std::string_view op) {
    require_not_bool(a, op);
    require_not_bool(b, op);
}

// 比较的契约 dtype 是 BOOL。仍把结果写成 INT8 的后端内核在这里统一规范化，因此
// 调用方（以及测试）看不到后端之间的差异。
Tensor as_bool(Tensor r) {
    if (r.dtype() == DataType::BOOL) return r;
    return ops::Typecast(static_cast<const Tensor&>(r), DataType::BOOL);
}

// 原地算术不能改变张量的 dtype。类 Python float 的标量会让整型/BOOL 输入想要浮点
// 结果，所以直接拒绝，而不是悄悄截断（PyTorch 在同样情况下也报错）。
void require_inplace_scalar_dtype(const Tensor& a, std::string_view op) {
    require_contiguous(a, op);
    require_not_bool(a, op);
    if (scalar_promote_type(a.dtype()) != a.dtype())
        throw std::runtime_error(std::format(
            "{}_: a float scalar would promote {} to {}; cast the tensor first",
            op, dtype_to_string(a.dtype()), dtype_to_string(scalar_promote_type(a.dtype()))));
}

// 非原地标量算术遵循 PyTorch 的弱标量提升：浮点张量保持自身 dtype，整型/BOOL
// 张量采用默认浮点 dtype。
template <class Fn>
Tensor scalar_arith_promote(const Tensor& a, float b, Fn&& fn) {
    require_not_bool(a, "scalar arithmetic");
    const DataType rt = scalar_promote_type(a.dtype());
    if (rt == a.dtype()) return fn(a, b);
    Tensor promoted = ops::Typecast(a, rt);
    return fn(promoted, b);
}

// 负轴从末尾计数（PyTorch）。0-dim 输入没有任何轴。
int normalize_axis(int axis, size_t ndim, std::string_view op) {
    const int n = static_cast<int>(ndim);
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n)
        throw std::runtime_error(std::format(
            "{}: axis is out of range for a {}D tensor", op, n));
    return axis;
}

// 单轴归约产生的形状：该轴被移除，或保留为长度 1。1-D 输入得到 {}（rank 0），
// 与全归约的结果形状一致。
std::vector<int64_t> reduced_shape(std::span<const int64_t> shape, int axis, bool keepdim) {
    std::vector<int64_t> out;
    out.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int>(i) == axis) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

// 后端自己构造归约后的形状，并在去掉该轴会变成 rank 0 时回退到 {1}。在这里统一
// 规范化，使 0-dim 结果在各后端拼法一致（keepdim 只是对新分配连续结果的纯 reshape，
// 内核无需知道它）。
Tensor with_reduced_shape(Tensor result, std::span<const int64_t> in_shape, int axis,
                          bool keepdim) {
    std::vector<int64_t> want = reduced_shape(in_shape, axis, keepdim);
    if (result.shape().size() != want.size() ||
        !std::equal(want.begin(), want.end(), result.shape().begin())) {
        result = result.reshape(std::move(want));
    }
    return result;
}

// 对所有轴归约：展平（连续时是视图）后归约 0 轴。这样各后端的单轴内核仍是唯一的
// 归约实现，全归约的 dtype 规则不会与单轴归约产生漂移。
const Tensor& flatten_for_reduce(const Tensor& t, Tensor& storage) {
    const Tensor& lin = linearize(t, storage);
    storage = lin;
    storage = storage.reshape({static_cast<int64_t>(lin.numel())});
    return storage;
}

// ===== FR-1：空张量（numel == 0）支持 =====
// 空张量是真正的张量：它有形状、dtype 和设备，只是没有元素。它与默认构造的 Tensor
// 不同——后者根本没有存储，任何算子都会拒绝。
void require_live(const Tensor& t) {
    if (!t.get_impl())
        throw std::runtime_error("tensor has no storage (default-constructed Tensor)");
}

bool is_empty(const Tensor& t) { return t.numel() == 0; }

// 空轴的 mean/var/std 是 0/0；规定答案为 NaN。
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// 形状的元素数；与 calc_numel 不同，它接受长度为 0 的维。
size_t numel_of(std::span<const int64_t> shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    return n;
}

// 无元素的结果：不跑任何内核，因此后端内核无需空输入分支。形状/dtype/设备仍按真实
// 契约给出。
Tensor empty_on(const Tensor& like, std::span<const int64_t> shape, DataType dtype) {
    ops::DeviceGuard guard(like.device_id());
    return Tensor(std::vector<int64_t>(shape.begin(), shape.end()), dtype, like.device());
}

// 不经过归约内核、直接用空折叠的恒等元（sum -> 0、prod -> 1、norm -> 0）或 NaN
// （mean/var/std）算出的结果。
Tensor filled_on(const Tensor& like, std::span<const int64_t> shape, DataType dtype,
                 float value) {
    std::vector<int64_t> s(shape.begin(), shape.end());
    return execute_on<ops::FillImpl>(like.device_id(), s, dtype, value);
}

// 对应 ufunc 框架的 result_type 契约，供没有实例化 Op 结构体的空路径使用：恒等类
// 算子保持 dtype，其余算子把整型输入提升到 FLOAT32（整型张量的 sin/exp/...）。
DataType elementwise_result_type(DataType in, bool identity) {
    return identity ? in : (is_integral_dtype(in) ? DataType::FLOAT32 : in);
}

// 校验 NumPy 广播并返回输出形状，因此空输入与非空输入抛出完全相同的形状错误。
DimArray broadcast_shape_or_throw(const Tensor& a, const Tensor& b, std::string_view op) {
    DimArray out;
    if (!ops::ufunc::broadcast_shapes(a.shape(), b.shape(), out))
        throw std::runtime_error(std::format("{}: shapes are not broadcastable", op));
    return out;
}

DimArray broadcast_shape_or_throw(const Tensor& a, const Tensor& b, const Tensor& c,
                                  std::string_view op) {
    // 先折叠前两个，再把结果与第三个折叠。
    DimArray out;
    if (!ops::ufunc::broadcast_shapes(a.shape(), b.shape(), out))
        throw std::runtime_error(std::format("{}: shapes are not broadcastable", op));
    DimArray out2;
    if (!ops::ufunc::broadcast_shapes(std::span<const int64_t>(out), c.shape(), out2))
        throw std::runtime_error(std::format("{}: shapes are not broadcastable", op));
    return out2;
}

// 二元逐元素算子的共享空路径：校验并分配广播结果，不跑内核。`out` 已设置时
// 返回 true。
bool empty_binary_result(const Tensor& a, const Tensor& b, std::string_view op, Tensor& out) {
    if (!is_empty(a) && !is_empty(b)) return false;
    const DimArray shape = broadcast_shape_or_throw(a, b, op);
    out = empty_on(a, shape, compute_type(a.dtype(), b.dtype()));
    return true;
}

// 即使没有元素可写，out= 的契约（广播形状、提升后的 dtype、连续 dst）依然成立。
void check_empty_binary_dst(const Tensor& a, const Tensor& b, const Tensor& dst,
                            std::string_view op) {
    const DimArray shape = broadcast_shape_or_throw(a, b, op);
    if (!ops::ufunc::same_shape(dst.shape(), std::span<const int64_t>(shape)))
        throw std::runtime_error(std::format("{}: dst shape must equal the broadcast result shape", op));
    if (dst.dtype() != compute_type(a.dtype(), b.dtype()))
        throw std::runtime_error(std::format("{}: dst dtype must equal the promoted result dtype", op));
    require_contiguous(dst, op);
}

}  // namespace

namespace ops {

void println(Tensor& a) {
    require_live(a);
    if (a.device() == Device::CPU) {
        if (a.is_contiguous()) {
            PrintlnImpl<Device::CPU>::execute(a);
        } else {
            Tensor temp = a.clone();
            PrintlnImpl<Device::CPU>::execute(temp);
        }
    } else {
        Tensor temp = a.clone();
        temp.to_host();
        PrintlnImpl<Device::CPU>::execute(temp);
    }
    std::cout << std::format("Tensor dtype: {} | Tensor device: {}",
                             dtype_to_string(a.dtype()), device_to_string(a.device())) << std::endl;
}

void println(Tensor&& a) {
    require_live(a);
    a.to_contiguous();
    Device device = a.device();
    if (a.device() == Device::CPU) {
        PrintlnImpl<Device::CPU>::execute(a);
    } else {
        a.to_host();
        PrintlnImpl<Device::CPU>::execute(a);
    }
    std::cout << std::format("Tensor dtype: {} | Tensor device: {}",
                             dtype_to_string(a.dtype()), device_to_string(device)) << std::endl;
}

void export_csv(Tensor& a, const std::string& path) {
    require_live(a);
    if (a.shape().size() > 2)
        throw std::runtime_error(std::format("export_csv: only 1D or 2D tensors are supported, got {}D", a.shape().size()));
    if (a.device() == Device::CPU) {
        if (a.is_contiguous()) {
            ExportCsvImpl<Device::CPU>::execute(a, path);
        } else {
            Tensor temp = a.clone();
            ExportCsvImpl<Device::CPU>::execute(temp, path);
        }
    } else {
        Tensor temp = a.clone();
        temp.to_host();
        ExportCsvImpl<Device::CPU>::execute(temp, path);
    }
    std::cout << std::format("Exported to '{}' | Tensor dtype: {} | Tensor device: {}",
                             path, dtype_to_string(a.dtype()), device_to_string(a.device())) << std::endl;
}

void export_csv(Tensor&& a, const std::string& path) {
    require_live(a);
    if (a.shape().size() > 2)
        throw std::runtime_error(std::format("export_csv: only 1D or 2D tensors are supported, got {}D", a.shape().size()));
    a.to_contiguous();
    Device device = a.device();
    if (a.device() == Device::CPU) {
        ExportCsvImpl<Device::CPU>::execute(a, path);
    } else {
        a.to_host();
        ExportCsvImpl<Device::CPU>::execute(a, path);
    }
    std::cout << std::format("Exported to '{}' | Tensor dtype: {} | Tensor device: {}",
                             path, dtype_to_string(a.dtype()), device_to_string(device)) << std::endl;
}

Tensor Ones(const std::vector<int64_t>& shape, DataType dtype, Device device) {
    return execute_on<OnesImpl>(current_id(device), shape, dtype);
}

// TensorOptions / DeviceId 形式直接解析实例、不做线程私有查找，因此任何线程都能
// 命名 CUDA:1（FR-7）。
Tensor Ones(const std::vector<int64_t>& shape, const TensorOptions& options) {
    return execute_on<OnesImpl>(options.resolved_id(), shape, options.dtype);
}
Tensor Zeros(const std::vector<int64_t>& shape, const TensorOptions& options) {
    return execute_on<ZerosImpl>(options.resolved_id(), shape, options.dtype);
}
Tensor Fill(const std::vector<int64_t>& shape, float val, const TensorOptions& options) {
    return execute_on<FillImpl>(options.resolved_id(), shape, options.dtype, val);
}
Tensor Random(const std::vector<int64_t>& shape, float min, float max, const TensorOptions& options) {
    if (is_bool_dtype(options.dtype))
        throw std::runtime_error("random: BOOL is not supported; use a comparison or cast instead");
    return execute_on<RandomImpl>(options.resolved_id(), shape, options.dtype, min, max);
}
Tensor Randn(const std::vector<int64_t>& shape, float mean, float std, const TensorOptions& options) {
    if (is_bool_dtype(options.dtype))
        throw std::runtime_error("randn: BOOL is not supported; use a comparison or cast instead");
    if (std < 0.0f) throw std::runtime_error("randn: std must be non-negative");
    // RNG 只保留一处：在宿主抽样，再把整块交给目标设备。同一 seed 在各后端取值相同。
    Tensor t = execute_on<RandnImpl>(DeviceId{Device::CPU, 0}, shape, options.dtype, mean, std);
    t.to_device(options.resolved_id());
    return t;
}
Tensor Ones(const std::vector<int64_t>& shape, DataType dtype, DeviceId device) {
    return execute_on<OnesImpl>(device, shape, dtype);
}
Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, DeviceId device) {
    return execute_on<ZerosImpl>(device, shape, dtype);
}
Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float val, DeviceId device) {
    return execute_on<FillImpl>(device, shape, dtype, val);
}
Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, DeviceId device) {
    if (is_bool_dtype(dtype))
        throw std::runtime_error("random: BOOL is not supported; use a comparison or cast instead");
    return execute_on<RandomImpl>(device, shape, dtype, min, max);
}

Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, Device device) {
    return execute_on<ZerosImpl>(current_id(device), shape, dtype);
}

Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float val, Device device) {
    return execute_on<FillImpl>(current_id(device), shape, dtype, val);
}

Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, Device device) {
    if (is_bool_dtype(dtype))
        throw std::runtime_error("random: BOOL is not supported; use a comparison or cast instead");
    return execute_on<RandomImpl>(current_id(device), shape, dtype, min, max);
}

Tensor Randn(const std::vector<int64_t>& shape, DataType dtype, float mean, float std, Device device) {
    if (is_bool_dtype(dtype))
        throw std::runtime_error("randn: BOOL is not supported; use a comparison or cast instead");
    if (std < 0.0f) throw std::runtime_error("randn: std must be non-negative");
    const DeviceId target = current_id(device);
    Tensor t = execute_on<RandnImpl>(DeviceId{Device::CPU, 0}, shape, dtype, mean, std);
    t.to_device(target);
    return t;
}

Tensor Arange(float start, float stop, float step, DataType dtype) {
    if (step == 0.0f) throw std::runtime_error("arange: step cannot be zero");
    if ((step > 0.0f && start >= stop) || (step < 0.0f && start <= stop))
        throw std::runtime_error("arange: empty range for the given step sign");
    return ArangeImpl<Device::CPU>::execute(start, stop, step, dtype);
}

Tensor Linspace(float start, float stop, int64_t steps, DataType dtype) {
    if (steps <= 0) throw std::runtime_error("linspace: steps must be positive");
    return LinspaceImpl<Device::CPU>::execute(start, stop, steps, dtype);
}

Tensor Eye(int64_t n, DataType dtype) {
    if (n <= 0) throw std::runtime_error("eye: n must be positive");
    return EyeImpl<Device::CPU>::execute(n, n, dtype);
}

Tensor Eye(int64_t n, int64_t m, DataType dtype) {
    if (n <= 0 || m <= 0) throw std::runtime_error("eye: dimensions must be positive");
    return EyeImpl<Device::CPU>::execute(n, m, dtype);
}

// 生成器的设备形式（FR-10）：值在宿主排布后一次性传输。确定性算子因此精确；
// randn 走唯一的宿主 RNG，所以同一 seed 在各后端含义相同。
Tensor Arange(float start, float stop, float step, const TensorOptions& options) {
    Tensor t = Arange(start, stop, step, options.dtype);
    t.to_device(options.resolved_id());
    return t;
}

Tensor Linspace(float start, float stop, int64_t steps, const TensorOptions& options) {
    Tensor t = Linspace(start, stop, steps, options.dtype);
    t.to_device(options.resolved_id());
    return t;
}

Tensor Eye(int64_t n, const TensorOptions& options) {
    Tensor t = Eye(n, options.dtype);
    t.to_device(options.resolved_id());
    return t;
}

Tensor Eye(int64_t n, int64_t m, const TensorOptions& options) {
    Tensor t = Eye(n, m, options.dtype);
    t.to_device(options.resolved_id());
    return t;
}

Tensor Slice(const Tensor& t, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    if (ranges.empty())  throw std::runtime_error("Slice ranges cannot be empty");
    if (ranges.size() != t.shape().size())   throw std::runtime_error("Slice ranges size must be equal to tensor shape size");
    for (size_t i = 0; i < ranges.size(); i++) {
        if (ranges[i].first > ranges[i].second)
            throw std::runtime_error("first must be less than second");
        if (ranges[i].first < 0 || ranges[i].first > ranges[i].second)
            throw std::runtime_error("first must be >= 0 and <= second");
        if (ranges[i].second > t.shape(i))
            throw std::runtime_error(std::format("second must be less than {}", t.shape(i)));
    }
    return execute_on<SliceImpl>(t.device_id(), t, ranges);
}

void Add(Tensor& a, float b) {
    require_inplace_scalar_dtype(a, "add");
    if (a.numel() == 0 || b == 0.0f) return;
    execute_on<AddImpl>(a.device_id(), a, b);
}
void Sub(Tensor& a, float b) {
    require_inplace_scalar_dtype(a, "sub");
    if (a.numel() == 0 || b == 0.0f) return;
    execute_on<SubImpl>(a.device_id(), a, b);
}
void Mul(Tensor& a, float b) {
    require_inplace_scalar_dtype(a, "mul");
    if (a.numel() == 0 || b == 1.0f) return;
    execute_on<MulImpl>(a.device_id(), a, b);
}
void Div(Tensor& a, float b) {
    require_inplace_scalar_dtype(a, "div");
    if (a.numel() == 0 || b == 1.0f) return;
    if (b == 0.0f) throw std::runtime_error("div: Cann't division by zero");
    execute_on<DivImpl>(a.device_id(), a, b);
}

Tensor Add(const Tensor& a, float b) {
    return scalar_arith_promote(a, b, [](const Tensor& x, float v) {
        if (x.numel() == 0 || v == 0.0f) return x.clone();
        return execute_on<AddImpl>(x.device_id(), x, v);
    });
}
Tensor Sub(const Tensor& a, float b) {
    return scalar_arith_promote(a, b, [](const Tensor& x, float v) {
        if (x.numel() == 0 || v == 0.0f) return x.clone();
        return execute_on<SubImpl>(x.device_id(), x, v);
    });
}
Tensor Mul(const Tensor& a, float b) {
    return scalar_arith_promote(a, b, [](const Tensor& x, float v) {
        if (x.numel() == 0 || v == 1.0f) return x.clone();
        return execute_on<MulImpl>(x.device_id(), x, v);
    });
}
Tensor Div(const Tensor& a, float b) {
    if (b == 0.0f) throw std::runtime_error("Division by zero");
    return scalar_arith_promote(a, b, [](const Tensor& x, float v) {
        if (x.numel() == 0 || v == 1.0f) return x.clone();
        return execute_on<DivImpl>(x.device_id(), x, v);
    });
}

Tensor Add(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "add");
    Tensor out;
    if (empty_binary_result(a, b, "add", out)) return out;

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<AddImpl>(a.device_id(), a, b);
}
Tensor Sub(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "sub");
    Tensor out;
    if (empty_binary_result(a, b, "sub", out)) return out;

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<SubImpl>(a.device_id(), a, b);
}
Tensor Mul(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "mul");
    Tensor out;
    if (empty_binary_result(a, b, "mul", out)) return out;

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<MulImpl>(a.device_id(), a, b);
}
Tensor Div(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Device mismatch!");
    require_not_bool(a, b, "div");
    Tensor out;
    if (empty_binary_result(a, b, "div", out)) return out;

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<DivImpl>(a.device_id(), a, b);
}

void Add(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device_id() != b.device_id() || dst.device_id() != a.device_id() || dst.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "add");
    require_not_bool(dst, "add");
    if (is_empty(a) || is_empty(b)) {
        check_empty_binary_dst(a, b, dst, "add");
        return;
    }

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<AddImpl>(a.device_id(), a, b, dst);
}
void Sub(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device_id() != b.device_id() || dst.device_id() != a.device_id() || dst.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "sub");
    require_not_bool(dst, "sub");
    if (is_empty(a) || is_empty(b)) {
        check_empty_binary_dst(a, b, dst, "sub");
        return;
    }

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<SubImpl>(a.device_id(), a, b, dst);
}
void Mul(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device_id() != b.device_id() || dst.device_id() != a.device_id() || dst.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "mul");
    require_not_bool(dst, "mul");
    if (is_empty(a) || is_empty(b)) {
        check_empty_binary_dst(a, b, dst, "mul");
        return;
    }

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<MulImpl>(a.device_id(), a, b, dst);
}
void Div(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device_id() != b.device_id() || dst.device_id() != a.device_id() || dst.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    require_not_bool(a, b, "div");
    require_not_bool(dst, "div");
    if (is_empty(a) || is_empty(b)) {
        check_empty_binary_dst(a, b, dst, "div");
        return;
    }

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<DivImpl>(a.device_id(), a, b, dst);
}

// ===== FR-9：matmul =====
// PyTorch 的 matmul 语义：最后两维是矩阵，1-D 操作数是行（左侧）或列（右侧）向量，
// 矩阵维之前的一切都是按右对齐广播的 batch。每个后端内核都只处理连续的
// [batch, rows, inner] x [batch, inner, cols]，所以 N-D 情况在这里折叠成那种形态，
// 而不是分别教三个 GPU 后端。
namespace {

struct MatmulDims {
    std::vector<int64_t> batch;
    int64_t rows = 1;
    int64_t cols = 1;
    int64_t inner = 0;
    bool a_vector = false;
    bool b_vector = false;
};

int64_t broadcast_dim(int64_t lhs, int64_t rhs, std::string_view op) {
    if (lhs == rhs || rhs == 1) return lhs;
    if (lhs == 1) return rhs;
    throw std::runtime_error(std::format(
        "{}: batch dimensions do not broadcast ({} vs {})", op, lhs, rhs));
}

MatmulDims matmul_dims(const Tensor& a, const Tensor& b, std::string_view op) {
    const size_t ad = a.shape().size();
    const size_t bd = b.shape().size();
    if (ad == 0 || bd == 0)
        throw std::runtime_error(std::format(
            "{}: operands must have at least one dimension", op));
    MatmulDims d;
    d.a_vector = ad == 1;
    d.b_vector = bd == 1;
    d.rows = d.a_vector ? 1 : a.shape(ad - 2);
    d.inner = a.shape(ad - 1);
    d.cols = d.b_vector ? 1 : b.shape(bd - 1);
    if (b.shape(bd - 2) != d.inner)
        throw std::runtime_error(std::format(
            "{}: inner dimensions do not match ({} vs {})", op, d.inner, b.shape(bd - 2)));

    const size_t an = ad > 2 ? ad - 2 : 0;
    const size_t bn = bd > 2 ? bd - 2 : 0;
    const size_t nb = std::max(an, bn);
    d.batch.assign(nb, 1);
    for (size_t i = 0; i < nb; ++i) {
        // i 从尾部 batch 维开始计数，两个操作数就按 NumPy 广播的方式对齐。
        int64_t dim = 1;
        if (i < an) dim = broadcast_dim(a.shape(an - 1 - i), dim, op);
        if (i < bn) dim = broadcast_dim(b.shape(bn - 1 - i), dim, op);
        d.batch[nb - 1 - i] = dim;
    }
    return d;
}

// 把操作数按右对齐广播进连续的 [batch, rows, cols] buffer。若既没有重复的 batch 维
// 也不是跨步布局，每一步都只是视图，操作数零拷贝直达内核。
Tensor matmul_operand(const Tensor& t, std::span<const int64_t> batch,
                      int64_t rows, int64_t cols) {
    Tensor r = t.shape().size() < 2 ? t.reshape({rows, cols}) : t;
    std::vector<int64_t> target(batch.begin(), batch.end());
    target.push_back(rows);
    target.push_back(cols);
    r = r.expand(std::move(target));
    r = r.contiguous();
    // 连续但带 offset 的操作数仍是别人 buffer 里的视图；Vulkan 内核从 buffer 基址
    // 寻址，所以先把它折叠掉。
    if (needs_offset_free_input(r)) r = r.clone();
    return r.reshape({static_cast<int64_t>(numel_of(batch)), rows, cols});
}

}  // namespace

Tensor MatMul(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id())
        throw std::runtime_error("matmul: operands must live on the same device");
    require_live(a);
    require_live(b);
    require_not_bool(a, b, "matmul");
    const MatmulDims d = matmul_dims(a, b, "matmul");

    // 被提升的向量维要从结果里去掉，而不是保留成长度 1：vec @ vec 是 rank 0，
    // vec @ matrix 是 rank 1。
    const DataType out_type = compute_type(a.dtype(), b.dtype());
    std::vector<int64_t> out_shape = d.batch;
    if (!d.a_vector) out_shape.push_back(d.rows);
    if (!d.b_vector) out_shape.push_back(d.cols);

    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    // 对空维收缩是空点积：没有乘积的和是 0，不是 NaN。
    if (d.inner == 0) return filled_on(a, out_shape, out_type, 0.0f);

    const Tensor A = matmul_operand(a, d.batch, d.rows, d.inner);
    const Tensor B = matmul_operand(b, d.batch, d.inner, d.cols);
    Tensor out = execute_on<MatMulImpl>(A.device_id(), A, B);
    // 内核按 [batch, rows, cols] 作答；只需去掉被提升的向量维。
    if (out.shape().size() != out_shape.size() ||
        !std::equal(out_shape.begin(), out_shape.end(), out.shape().begin()))
        out = out.reshape(std::move(out_shape));
    return out;
}

Tensor Bmm(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 3 || b.shape().size() != 3)
        throw std::runtime_error(std::format(
            "bmm: both operands must be 3-D, got {}D and {}D",
            a.shape().size(), b.shape().size()));
    if (a.shape(0) != b.shape(0))
        throw std::runtime_error(std::format(
            "bmm: batch sizes must match ({} vs {})", a.shape(0), b.shape(0)));
    return MatMul(a, b);
}

Tensor Addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
             float beta, float alpha) {
    if (input.device_id() != mat1.device_id() || input.device_id() != mat2.device_id())
        throw std::runtime_error("addmm: all three operands must live on the same device");
    // out = beta * input + alpha * (mat1 @ mat2)。`input` 广播到 matmul 结果形状。
    // beta == 0 时完全跳过它，调用方不想读的加数就绝不会被读（PyTorch 同样处理）。
    // alpha 只缩放乘积，所以在加上加数之前生效。
    Tensor out = MatMul(mat1, mat2);
    if (alpha != 1.0f) out = ops::Mul(static_cast<const Tensor&>(out), alpha);
    if (beta != 0.0f) out = ops::Add(out, beta == 1.0f ? input : ops::Mul(input, beta));
    return out;
}

Tensor Linear(const Tensor& input, const Tensor& weight) {
    if (weight.shape().size() != 2)
        throw std::runtime_error(std::format(
            "linear: weight must be 2-D [out_features, in_features], got {}D",
            weight.shape().size()));
    // y = x @ weight^T，支持任意数量的前导 batch 维（同一套 N-D matmul 同时覆盖
    // [tokens, in] 与 [batch, tokens, in]）。
    return MatMul(input, ops::Transpose(weight, {1, 0}));
}

Tensor Linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    return ops::Add(Linear(input, weight), bias);
}

namespace {

// 卷积的入口校验与形状推导集中在这里：四个后端实现拿到的已经是 4-D 连续 NCHW，
// 只做下标算术。stride/padding 是标量广播到两维，dilation 与 groups 不在契约内。
Tensor conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                      int64_t stride, int64_t padding) {
    const auto in_shape = input.shape();
    const auto w_shape = weight.shape();
    const size_t in_rank = in_shape.size();
    if (in_rank != 3 && in_rank != 4)
        throw std::runtime_error(std::format(
            "conv2d: input must be 3-D [C, H, W] or 4-D [N, C, H, W], got {}D", in_rank));
    if (w_shape.size() != 4)
        throw std::runtime_error(std::format(
            "conv2d: weight must be 4-D [Cout, C, kh, kw], got {}D", w_shape.size()));
    if (stride < 1) throw std::runtime_error("conv2d: stride must be at least 1");
    if (padding < 0) throw std::runtime_error("conv2d: padding must not be negative");

    // 与 PyTorch 一致：三者的 dtype 与设备必须完全一致，不做隐式提升。
    if (weight.dtype() != input.dtype())
        throw std::runtime_error(std::format(
            "conv2d: input and weight dtypes must match ({} vs {})",
            dtype_to_string(input.dtype()), dtype_to_string(weight.dtype())));
    if (weight.device_id() != input.device_id())
        throw std::runtime_error("conv2d: input and weight must live on the same device");
    if (!is_floating_dtype(input.dtype()))
        throw std::runtime_error(std::format("conv2d: dtype must be floating point, got {}",
                                             dtype_to_string(input.dtype())));

    const int64_t channels = in_shape[in_rank - 3];
    const int64_t in_h = in_shape[in_rank - 2];
    const int64_t in_w = in_shape[in_rank - 1];
    const int64_t batch = in_rank == 4 ? in_shape[0] : 1;
    const int64_t out_ch = w_shape[0];
    const int64_t kh = w_shape[2];
    const int64_t kw = w_shape[3];

    if (w_shape[1] != channels)
        throw std::runtime_error(std::format(
            "conv2d: weight expects {} input channels but the input has {}",
            w_shape[1], channels));
    if (kh <= 0 || kw <= 0)
        throw std::runtime_error("conv2d: kernel size must be positive");

    const int64_t out_h = (in_h + 2 * padding - kh) / stride + 1;
    const int64_t out_w = (in_w + 2 * padding - kw) / stride + 1;
    if (out_h <= 0 || out_w <= 0)
        throw std::runtime_error(std::format(
            "conv2d: {0}x{1} input with {2}x{3} kernel, stride {4}, padding {5} "
            "yields a non-positive output size",
            in_h, in_w, kh, kw, stride, padding));

    const Conv2dShape shape{.batch = batch,       .in_ch = channels, .in_h = in_h,
                            .in_w = in_w,         .out_ch = out_ch,  .kh = kh,
                            .kw = kw,             .stride = stride,  .padding = padding,
                            .out_h = out_h,       .out_w = out_w};

    std::vector<int64_t> out_shape{batch, out_ch, out_h, out_w};
    if (batch == 0 || out_ch == 0)
        return empty_on(input, out_shape, input.dtype());

    // 内核线性寻址，所以吃连续、且不带视图 offset 的操作数。
    Tensor in_hold, w_hold, b_hold;
    Tensor plain = in_rank == 3 ? input.unsqueeze(0) : input;
    const Tensor& src = linearize(plain, in_hold);
    const Tensor& w = linearize(weight, w_hold);

    const Tensor* bias_arg = nullptr;
    if (bias != nullptr) {
        if (bias->numel() != 0) {
            if (bias->dtype() != input.dtype())
                throw std::runtime_error(std::format(
                    "conv2d: bias dtype must match the input ({} vs {})",
                    dtype_to_string(bias->dtype()), dtype_to_string(input.dtype())));
            if (bias->device_id() != input.device_id())
                throw std::runtime_error("conv2d: bias must live on the same device as the input");
            if (bias->numel() != out_ch)
                throw std::runtime_error(std::format(
                    "conv2d: bias must have {} elements, got {}", out_ch, bias->numel()));
            bias_arg = &linearize(*bias, b_hold);
        }
    }

    Tensor out = execute_on<Conv2dImpl>(src.device_id(), src, w, bias_arg, shape);
    if (in_rank == 3) out = out.reshape({out_ch, out_h, out_w});
    return out;
}

}  // namespace

Tensor Conv2d(const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding) {
    return conv2d_forward(input, weight, nullptr, stride, padding);
}

Tensor Conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t stride,
              int64_t padding) {
    return conv2d_forward(input, weight, &bias, stride, padding);
}

Tensor Abs(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "abs");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), true));
    return execute_on<AbsImpl>(a.device_id(), a);
}
Tensor Sin(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "sin");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<SinImpl>(a.device_id(), a);
}
Tensor Cos(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "cos");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<CosImpl>(a.device_id(), a);
}
Tensor Tan(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "tan");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<TanImpl>(a.device_id(), a);
}
Tensor Exp(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "exp");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<ExpImpl>(a.device_id(), a);
}
Tensor Relu(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "relu");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), true));
    return execute_on<ReluImpl>(a.device_id(), a);
}
Tensor Silu(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "silu");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<SiluImpl>(a.device_id(), a);
}
Tensor Tanh(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "tanh");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<TanhImpl>(a.device_id(), a);
}
Tensor Sqrt(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "sqrt");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    if (ops::Any((a < 0), 1)) throw std::runtime_error("tensor must be non-negative");
    return execute_on<SqrtImpl>(a.device_id(), a);
}
Tensor Sigmoid(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "sigmoid");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<SigmoidImpl>(a.device_id(), a);
}
Tensor Pow(const Tensor& a, float val) {
    require_live(a);
    require_not_bool(a, "pow");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<PowImpl>(a.device_id(), a, val);
}
Tensor Log(const Tensor& a, float val) {
    if (val < 0 || val == 1) throw std::runtime_error("The base cannot be less than or equal to 0, and it cannot be 1");
    require_live(a);
    require_not_bool(a, "log");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    if (ops::Any((a < 0), 1)) throw std::runtime_error("log: tensor must be non-negative");
    return execute_on<LogImpl>(a.device_id(), a, val);
}
Tensor Softmax(const Tensor& a, int axis) {
    require_live(a);
    require_not_bool(a, "softmax");
    if (std::abs(axis) >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis out of range");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<SoftmaxImpl>(A.device_id(), A, axis);
}
// ===== FR-10：用组合实现设备覆盖 =====
namespace {

// 归一化/softmax 族在 CPU 上有专用内核；设备后端用它们都已实现的原子算子
// （Sum/Mean/Max/Exp/Log/Sqrt/Abs/Sub/Mul/Div）拼出同一个公式来运行。每一步都是
// 设备算子，没有任何静默回退到宿主，两遍算法与 CPU 内核逐值一致。
Tensor promote_for_reduce(const Tensor& a, DataType rt) {
    return a.dtype() == rt ? a : Typecast(a, rt);
}

Tensor log_softmax_composed(const Tensor& a, int axis) {
    const Tensor x = promote_for_reduce(a, elementwise_result_type(a.dtype(), false));
    const Tensor m = Max(x, axis, true);
    const Tensor s = Sum(Exp(Sub(x, m)), axis, true);
    // log_softmax = x - m - log(Σ exp(x - m))，这里的 log 是以 e 为底的 Log。
    return Sub(Sub(x, m), Log(s, 2.718281828459045f));
}

Tensor var_composed(const Tensor& a, int axis, bool keepdim) {
    const Tensor x = promote_for_reduce(a, float_reduction_type(a.dtype()));
    const Tensor m = Mean(x, axis, true);
    const Tensor d = Sub(x, m);
    return Mean(Mul(d, d), axis, keepdim);
}

Tensor norm_composed(const Tensor& a, float p, int axis, bool keepdim) {
    const Tensor x = promote_for_reduce(a, float_reduction_type(a.dtype()));
    if (p == 1.0f) return Sum(Abs(x), axis, keepdim);
    return Sqrt(Sum(Mul(x, x), axis, keepdim));
}

// LayerNorm 把尾部块 [axis, end) 作为整体归一化，因此组合形式在 [outer, norm_size]
// 视图上计算，weight/bias 广播到最后一维。
Tensor layer_norm_composed(const Tensor& a, const Tensor& weight, const Tensor& bias,
                           int64_t outer, int64_t norm_size, float eps, DataType rt) {
    const Tensor x = promote_for_reduce(a, rt).reshape({outer, norm_size});
    const Tensor d = Sub(x, Mean(x, 1, true));
    const Tensor var = Mean(Mul(d, d), 1, true);
    Tensor y = Div(d, Sqrt(Add(var, eps)));
    y = Add(Mul(y, weight), bias);
    return y.reshape(std::vector<int64_t>(a.shape().begin(), a.shape().end()));
}

Tensor rms_norm_composed(const Tensor& a, const Tensor& weight,
                         int64_t outer, int64_t norm_size, float eps, DataType rt) {
    const Tensor x = promote_for_reduce(a, rt).reshape({outer, norm_size});
    const Tensor ms = Mean(Mul(x, x), 1, true);
    Tensor y = Div(x, Sqrt(Add(ms, eps)));
    y = Mul(y, weight);
    return y.reshape(std::vector<int64_t>(a.shape().begin(), a.shape().end()));
}

}  // namespace

Tensor LogSoftmax(const Tensor& a, int axis) {
    require_live(a);
    require_not_bool(a, "log_softmax");
    if (std::abs(axis) >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis out of range");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    if (a.device() != Device::CPU) return log_softmax_composed(a, axis);
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return LogSoftmaxImpl<Device::CPU>::execute(A, axis);
}
Tensor Erf(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "erf");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), false));
    return execute_on<ErfImpl>(a.device_id(), a);
}
Tensor LayerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps) {
    require_live(a);
    require_not_bool(a, "layer_norm");
    require_not_bool(weight, "layer_norm");
    require_not_bool(bias, "layer_norm");
    if (eps <= 0.0f) throw std::runtime_error("layer_norm: eps must be positive");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis < 0 || axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("layer_norm: axis out of range");
    auto a_shape = a.shape();
    int dim = a_shape.size();
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];
    DataType res_type = is_integral_dtype(a.dtype()) ? DataType::FLOAT32 : a.dtype();
    if (weight.numel() != norm_size) throw std::runtime_error("layer_norm: weight numel must match normalized block size");
    if (bias.numel() != norm_size) throw std::runtime_error("layer_norm: bias numel must match normalized block size");
    if (weight.dtype() != res_type || bias.dtype() != res_type)
        throw std::runtime_error("layer_norm: weight/bias dtype must match result dtype");
    if (is_empty(a)) return empty_on(a, a.shape(), res_type);
    if (a.device() != Device::CPU) {
        int64_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= a_shape[i];
        return layer_norm_composed(a, weight, bias, outer, static_cast<int64_t>(norm_size),
                                   eps, res_type);
    }
    Tensor sa, sw, sb;
    const Tensor& A = linearize(a, sa);
    const Tensor& W = linearize(weight, sw);
    const Tensor& B = linearize(bias, sb);
    return LayerNormImpl<Device::CPU>::execute(A, W, B, axis, eps);
}
Tensor RmsNorm(const Tensor& a, const Tensor& weight, int axis, float eps) {
    require_live(a);
    require_not_bool(a, "rms_norm");
    require_not_bool(weight, "rms_norm");
    if (eps <= 0.0f) throw std::runtime_error("rms_norm: eps must be positive");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis < 0 || axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("rms_norm: axis out of range");
    auto a_shape = a.shape();
    int dim = static_cast<int>(a_shape.size());
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];
    DataType res_type = is_integral_dtype(a.dtype()) ? DataType::FLOAT32 : a.dtype();
    if (weight.numel() != norm_size) throw std::runtime_error("rms_norm: weight numel must match normalized block size");
    if (weight.dtype() != res_type)
        throw std::runtime_error("rms_norm: weight dtype must match result dtype");
    if (is_empty(a)) return empty_on(a, a.shape(), res_type);
    if (a.device() != Device::CPU) {
        int64_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= a_shape[i];
        return rms_norm_composed(a, weight, outer, static_cast<int64_t>(norm_size),
                                 eps, res_type);
    }
    Tensor sa, sw;
    const Tensor& A = linearize(a, sa);
    const Tensor& W = linearize(weight, sw);
    return RmsNormImpl<Device::CPU>::execute(A, W, axis, eps);
}
Tensor Clamp(const Tensor& a, float min, float max) {
    require_live(a);
    require_not_bool(a, "clamp");
    if (min > max) throw std::runtime_error("min must be less than max");
    if (is_empty(a)) return empty_on(a, a.shape(), elementwise_result_type(a.dtype(), true));
    return execute_on<ClampImpl>(a.device_id(), a, min, max);
}

void Zeros(Tensor& a) {
    require_contiguous(a, "zeros");
    require_live(a);
    if (is_empty(a)) return;
    if (a.device() != Device::CPU) {
        execute_on<FillImpl>(a.device_id(), a, 0.0f);
        return;
    }
    // 所有支持 dtype（int / float / float16 / bfloat16）的零表示都是全零 bit pattern，
    // memset 比走 dispatch_dtype 的循环快得多。
    std::memset(a.data(), 0, a.numel() * calc_dtype_size(a.dtype()));
}

void Fill(Tensor& a, float value) {
    require_contiguous(a, "fill");
    require_live(a);
    if (is_empty(a)) return;
    if (a.device() != Device::CPU) {
        execute_on<FillImpl>(a.device_id(), a, value);
        return;
    }
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(a.data());
        const T v = [&] {
            if constexpr (std::is_same_v<T, uint8_t>)
                return value != 0.0f ? uint8_t{1} : uint8_t{0};
            else
                return static_cast<T>(value);
        }();
        std::fill(p, p + a.numel(), v);
    });
}

void Ones(Tensor& a) {
    Fill(a, 1.0f);
}

void Abs(Tensor& a) {
    require_live(a);
    require_contiguous(a, "abs_");
    require_not_bool(a, "abs_");
    if (is_empty(a)) return;
    execute_on<AbsImpl>(a.device_id(), a);
}
void Sin(Tensor& a) {
    require_live(a);
    require_contiguous(a, "sin_");
    require_not_bool(a, "sin_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<SinImpl>(a.device_id(), a);
}
void Cos(Tensor& a) {
    require_live(a);
    require_contiguous(a, "cos_");
    require_not_bool(a, "cos_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<CosImpl>(a.device_id(), a);
}
void Tan(Tensor& a) {
    require_live(a);
    require_contiguous(a, "tan_");
    require_not_bool(a, "tan_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<TanImpl>(a.device_id(), a);
}
void Relu(Tensor& a) {
    require_live(a);
    require_contiguous(a, "relu_");
    require_not_bool(a, "relu_");
    if (is_empty(a)) return;
    execute_on<ReluImpl>(a.device_id(), a);
}
void Silu(Tensor& a) {
    require_live(a);
    require_contiguous(a, "silu_");
    require_not_bool(a, "silu_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<SiluImpl>(a.device_id(), a);
}
void Tanh(Tensor& a) {
    require_live(a);
    require_contiguous(a, "tanh_");
    require_not_bool(a, "tanh_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<TanhImpl>(a.device_id(), a);
}
void Pow(Tensor& a, float val) {
    require_live(a);
    require_contiguous(a, "pow_");
    require_not_bool(a, "pow_");
    if (is_empty(a)) return;
    execute_on<PowImpl>(a.device_id(), a, val);
}
void Sigmoid(Tensor& a) {
    require_live(a);
    require_contiguous(a, "sigmoid_");
    require_not_bool(a, "sigmoid_");
    if (is_integral_dtype(a.dtype()))
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    if (is_empty(a)) return;
    execute_on<SigmoidImpl>(a.device_id(), a);
}
void Clamp(Tensor& a, float min, float max) {
    require_live(a);
    require_contiguous(a, "clamp_");
    require_not_bool(a, "clamp_");
    if (min > max) throw std::runtime_error("min must be less than max");
    if (is_empty(a)) return;
    execute_on<ClampImpl>(a.device_id(), a, min, max);
}

Tensor Sum(const Tensor& a) {
    require_live(a);
    if (is_bool_dtype(a.dtype())) return Sum(Typecast(a, DataType::INT64));
    if (is_empty(a)) return filled_on(a, {}, reduction_type(a.dtype()), 0.0f);
    Tensor storage;
    return Sum(flatten_for_reduce(a, storage), 0, false);
}
Tensor Min(const Tensor& a) {
    require_live(a);
    if (is_bool_dtype(a.dtype()))
        return Typecast(Min(Typecast(a, DataType::INT8)), DataType::BOOL);
    if (is_empty(a)) throw std::runtime_error("min: an empty tensor has no minimum");
    Tensor storage;
    return Min(flatten_for_reduce(a, storage), 0, false);
}
Tensor Max(const Tensor& a) {
    require_live(a);
    if (is_bool_dtype(a.dtype()))
        return Typecast(Max(Typecast(a, DataType::INT8)), DataType::BOOL);
    if (is_empty(a)) throw std::runtime_error("max: an empty tensor has no maximum");
    Tensor storage;
    return Max(flatten_for_reduce(a, storage), 0, false);
}
Tensor Mean(const Tensor& a) {
    require_live(a);
    if (is_bool_dtype(a.dtype()))
        throw std::runtime_error("mean: BOOL is not supported; cast the tensor to a numeric dtype first");
    // 空的 mean 是 0/0：与 PyTorch、NumPy 一样规定为 NaN。
    if (is_empty(a)) return filled_on(a, {}, float_reduction_type(a.dtype()), kNaN);
    Tensor storage;
    return Mean(flatten_for_reduce(a, storage), 0, false);
}
Tensor Prod(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "prod");
    if (is_empty(a)) return filled_on(a, {}, reduction_type(a.dtype()), 1.0f);
    Tensor storage;
    return Prod(flatten_for_reduce(a, storage), 0, false);
}
Tensor Var(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "var");
    if (is_empty(a)) return filled_on(a, {}, float_reduction_type(a.dtype()), kNaN);
    Tensor storage;
    return Var(flatten_for_reduce(a, storage), 0, false);
}
Tensor Std(const Tensor& a) {
    require_live(a);
    require_not_bool(a, "std");
    if (is_empty(a)) return filled_on(a, {}, float_reduction_type(a.dtype()), kNaN);
    Tensor storage;
    return Std(flatten_for_reduce(a, storage), 0, false);
}
Tensor Norm(const Tensor& a, float p) {
    require_live(a);
    require_not_bool(a, "norm");
    if (p != 1.0f && p != 2.0f) throw std::runtime_error("norm: only p=1 or p=2 supported");
    if (is_empty(a)) return filled_on(a, {}, float_reduction_type(a.dtype()), 0.0f);
    Tensor storage;
    return Norm(flatten_for_reduce(a, storage), p, 0, false);
}

Tensor Sum(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    axis = normalize_axis(axis, a.dims(), "sum");
    if (is_bool_dtype(a.dtype())) return Sum(Typecast(a, DataType::INT64), axis, keepdim);
    const DataType out_type = reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, 0.0f);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(execute_on<SumImpl>(A.device_id(), A, axis), a.shape(), axis, keepdim);
}
Tensor Min(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    axis = normalize_axis(axis, a.dims(), "min");
    if (is_bool_dtype(a.dtype()))
        return Typecast(Min(Typecast(a, DataType::INT8), axis, keepdim), DataType::BOOL);
    if (a.shape(axis) == 0)
        throw std::runtime_error("min: reducing an empty axis has no minimum");
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, a.dtype());
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(execute_on<MinImpl>(A.device_id(), A, axis), a.shape(), axis, keepdim);
}
Tensor Max(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    axis = normalize_axis(axis, a.dims(), "max");
    if (is_bool_dtype(a.dtype()))
        return Typecast(Max(Typecast(a, DataType::INT8), axis, keepdim), DataType::BOOL);
    if (a.shape(axis) == 0)
        throw std::runtime_error("max: reducing an empty axis has no maximum");
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, a.dtype());
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(execute_on<MaxImpl>(A.device_id(), A, axis), a.shape(), axis, keepdim);
}
Tensor Mean(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    if (is_bool_dtype(a.dtype()))
        throw std::runtime_error("mean: BOOL is not supported; cast the tensor to a numeric dtype first");
    axis = normalize_axis(axis, a.dims(), "mean");
    const DataType out_type = float_reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, kNaN);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(execute_on<MeanImpl>(A.device_id(), A, axis), a.shape(), axis, keepdim);
}
Tensor Prod(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    require_not_bool(a, "prod");
    axis = normalize_axis(axis, a.dims(), "prod");
    const DataType out_type = reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, 1.0f);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(execute_on<ProdImpl>(A.device_id(), A, axis), a.shape(), axis, keepdim);
}
Tensor Var(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    require_not_bool(a, "var");
    axis = normalize_axis(axis, a.dims(), "var");
    const DataType out_type = float_reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, kNaN);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    if (a.device() != Device::CPU) return var_composed(a, axis, keepdim);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(VarImpl<Device::CPU>::execute(A, axis), a.shape(), axis, keepdim);
}
Tensor Std(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    require_not_bool(a, "std");
    axis = normalize_axis(axis, a.dims(), "std");
    const DataType out_type = float_reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, kNaN);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    if (a.device() != Device::CPU) return Sqrt(var_composed(a, axis, keepdim));
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(StdImpl<Device::CPU>::execute(A, axis), a.shape(), axis, keepdim);
}
Tensor Norm(const Tensor& a, float p, int axis, bool keepdim) {
    require_live(a);
    require_not_bool(a, "norm");
    if (p != 1.0f && p != 2.0f) throw std::runtime_error("norm: only p=1 or p=2 supported");
    axis = normalize_axis(axis, a.dims(), "norm");
    const DataType out_type = float_reduction_type(a.dtype());
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0) return filled_on(a, out_shape, out_type, 0.0f);
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, out_type);
    if (a.device() != Device::CPU) return norm_composed(a, p, axis, keepdim);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    return with_reduced_shape(NormImpl<Device::CPU>::execute(A, p, axis), a.shape(), axis, keepdim);
}
Tensor Cumsum(const Tensor& a, int axis) {
    require_live(a);
    require_not_bool(a, "cumsum");
    axis = normalize_axis(axis, a.dims(), "cumsum");
    if (is_empty(a)) return empty_on(a, a.shape(), a.dtype());
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<CumsumImpl>(A.device_id(), A, axis);
}

namespace {

std::vector<int> normalize_axes(std::span<const int64_t> dims, size_t ndim, std::string_view op) {
    std::vector<int> axes;
    axes.reserve(dims.size());
    for (int64_t d : dims) axes.push_back(normalize_axis(static_cast<int>(d), ndim, op));
    std::sort(axes.begin(), axes.end());
    for (size_t i = 1; i < axes.size(); ++i)
        if (axes[i] == axes[i - 1])
            throw std::runtime_error(std::format(
                "{}: axis {} appears more than once in the reduction list", op, axes[i]));
    return axes;
}

std::vector<int64_t> reduced_shape_multi(std::span<const int64_t> shape,
                                         const std::vector<int>& axes, bool keepdim) {
    std::vector<int64_t> out;
    out.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), static_cast<int>(i)) != axes.end()) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(shape[i]);
        }
    }
    return out;
}

// 每次折叠一轴，从最大轴开始，每一步都保留该轴，使其余下标仍然有效；最后由一次
// reshape 应用 keepdim。这样单轴内核就是所有轴数的唯一归约实现。
template <class Full, class Single>
Tensor reduce_axes(const Tensor& a, std::vector<int> axes, bool keepdim, Full&& full,
                   Single&& single) {
    if (axes.empty()) {
        // 空的 dim 列表表示“所有轴”。
        Tensor all = full(a);
        if (!keepdim) return all;
        std::vector<int64_t> ones(a.dims(), 1);
        return all.reshape(std::move(ones));
    }
    std::sort(axes.rbegin(), axes.rend());
    Tensor r = a;
    for (int ax : axes) r = single(r, ax, true);
    const std::vector<int64_t> want = reduced_shape_multi(a.shape(), axes, keepdim);
    if (r.dims() != want.size() || !std::equal(want.begin(), want.end(), r.shape().begin()))
        r = r.reshape(want);
    return r;
}

// 归约上的 `dtype=`：先由算子自身的规则决定如何累加，再把契约 dtype 应用到结果。
Tensor cast_reduction_result(Tensor r, DataType dtype, std::string_view op) {
    if (is_bool_dtype(dtype))
        throw std::runtime_error(std::format(
            "{}: BOOL is not a valid reduction output dtype; reduce to a numeric dtype", op));
    if (r.dtype() == dtype) return r;
    return r.to_type(dtype);
}

}  // namespace

Tensor Sum(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    return Sum(a, std::vector<int64_t>(dims), keepdim);
}
Tensor Sum(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim) {
    require_live(a);
    return reduce_axes(a, normalize_axes(dims, a.dims(), "sum"), keepdim,
                       [](const Tensor& x) { return Sum(x); },
                       [](const Tensor& x, int ax, bool kd) { return Sum(x, ax, kd); });
}
Tensor Min(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    return Min(a, std::vector<int64_t>(dims), keepdim);
}
Tensor Min(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim) {
    require_live(a);
    return reduce_axes(a, normalize_axes(dims, a.dims(), "min"), keepdim,
                       [](const Tensor& x) { return Min(x); },
                       [](const Tensor& x, int ax, bool kd) { return Min(x, ax, kd); });
}
Tensor Max(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    return Max(a, std::vector<int64_t>(dims), keepdim);
}
Tensor Max(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim) {
    require_live(a);
    return reduce_axes(a, normalize_axes(dims, a.dims(), "max"), keepdim,
                       [](const Tensor& x) { return Max(x); },
                       [](const Tensor& x, int ax, bool kd) { return Max(x, ax, kd); });
}
Tensor Mean(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    return Mean(a, std::vector<int64_t>(dims), keepdim);
}
Tensor Mean(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim) {
    require_live(a);
    return reduce_axes(a, normalize_axes(dims, a.dims(), "mean"), keepdim,
                       [](const Tensor& x) { return Mean(x); },
                       [](const Tensor& x, int ax, bool kd) { return Mean(x, ax, kd); });
}
Tensor Prod(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    return Prod(a, std::vector<int64_t>(dims), keepdim);
}
Tensor Prod(const Tensor& a, const std::vector<int64_t>& dims, bool keepdim) {
    require_live(a);
    return reduce_axes(a, normalize_axes(dims, a.dims(), "prod"), keepdim,
                       [](const Tensor& x) { return Prod(x); },
                       [](const Tensor& x, int ax, bool kd) { return Prod(x, ax, kd); });
}

// Var/Std/Norm 不能逐轴折叠（折叠的二阶矩不等于二阶矩的折叠；范数按范数组合，
// 不是一次折叠）。
Tensor Var(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    const std::vector<int64_t> v(dims);
    if (v.size() > 1)
        throw std::runtime_error("var: multi-axis reduction is not supported; reduce one axis at a time");
    require_live(a);
    if (v.empty()) return Var(a);
    return Var(a, normalize_axis(static_cast<int>(v[0]), a.dims(), "var"), keepdim);
}
Tensor Std(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim) {
    const std::vector<int64_t> v(dims);
    if (v.size() > 1)
        throw std::runtime_error("std: multi-axis reduction is not supported; reduce one axis at a time");
    require_live(a);
    if (v.empty()) return Std(a);
    return Std(a, normalize_axis(static_cast<int>(v[0]), a.dims(), "std"), keepdim);
}
Tensor Norm(const Tensor& a, float p, std::initializer_list<int64_t> dims, bool keepdim) {
    const std::vector<int64_t> v(dims);
    if (v.size() > 1)
        throw std::runtime_error("norm: multi-axis reduction is not supported; reduce one axis at a time");
    require_live(a);
    if (v.empty()) return Norm(a, p);
    return Norm(a, p, normalize_axis(static_cast<int>(v[0]), a.dims(), "norm"), keepdim);
}

Tensor Sum(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Sum(a), dtype, "sum");
}
Tensor Sum(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Sum(a, axis, keepdim), dtype, "sum");
}
Tensor Sum(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype) {
    return cast_reduction_result(Sum(a, dims, keepdim), dtype, "sum");
}
Tensor Min(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Min(a), dtype, "min");
}
Tensor Min(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Min(a, axis, keepdim), dtype, "min");
}
Tensor Min(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype) {
    return cast_reduction_result(Min(a, dims, keepdim), dtype, "min");
}
Tensor Max(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Max(a), dtype, "max");
}
Tensor Max(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Max(a, axis, keepdim), dtype, "max");
}
Tensor Max(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype) {
    return cast_reduction_result(Max(a, dims, keepdim), dtype, "max");
}
Tensor Mean(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Mean(a), dtype, "mean");
}
Tensor Mean(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Mean(a, axis, keepdim), dtype, "mean");
}
Tensor Mean(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype) {
    return cast_reduction_result(Mean(a, dims, keepdim), dtype, "mean");
}
Tensor Prod(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Prod(a), dtype, "prod");
}
Tensor Prod(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Prod(a, axis, keepdim), dtype, "prod");
}
Tensor Prod(const Tensor& a, std::initializer_list<int64_t> dims, bool keepdim, DataType dtype) {
    return cast_reduction_result(Prod(a, dims, keepdim), dtype, "prod");
}
Tensor Var(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Var(a), dtype, "var");
}
Tensor Var(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Var(a, axis, keepdim), dtype, "var");
}
Tensor Std(const Tensor& a, DataType dtype) {
    return cast_reduction_result(Std(a), dtype, "std");
}
Tensor Std(const Tensor& a, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Std(a, axis, keepdim), dtype, "std");
}
Tensor Norm(const Tensor& a, float p, DataType dtype) {
    return cast_reduction_result(Norm(a, p), dtype, "norm");
}
Tensor Norm(const Tensor& a, float p, int axis, bool keepdim, DataType dtype) {
    return cast_reduction_result(Norm(a, p, axis, keepdim), dtype, "norm");
}

namespace {

// 损失内核在宿主上按 float 累加，拿回来的是一个标量。这里把它包成与归约一致的
// 0 维张量：设备跟着输入走，dtype 走 mean 的规则（整型与 16 位浮点输入回到
// FLOAT32，FLOAT64 保持 FLOAT64）。损失目前只有 CPU 实现，所以这里的设备在
// 调用点已经由 require_cpu 限定为 CPU。
Tensor loss_result(float value, const Tensor& like) {
    return Fill(std::vector<int64_t>{}, float_reduction_type(like.dtype()), value,
                like.device_id());
}

}  // namespace

Tensor MseLoss(const Tensor& input, const Tensor& target) {
    if (input.device_id() != target.device_id()) throw std::runtime_error("Tensor Device mismatch!");
    if (input.numel() == 0) throw std::runtime_error("mse_loss: empty input");
    require_not_bool(input, "mse_loss");
    require_not_bool(target, "mse_loss");
    if (input.dtype() != target.dtype()) throw std::runtime_error("mse_loss: input and target must have same dtype");
    if (input.shape().size() != target.shape().size() || input.numel() != target.numel())
        throw std::runtime_error("mse_loss: input and target must have same shape");
    Tensor sa, sb;
    const Tensor& A = linearize(input, sa);
    const Tensor& B = linearize(target, sb);
    require_cpu(A, "mse_loss");
    return loss_result(MseLossImpl<Device::CPU>::execute(A, B), input);
}
Tensor NllLoss(const Tensor& input, const Tensor& target) {
    if (input.device_id() != target.device_id()) throw std::runtime_error("Tensor Device mismatch!");
    if (input.numel() == 0) throw std::runtime_error("nll_loss: empty input");
    require_not_bool(input, "nll_loss");
    if (input.shape().size() != 2) throw std::runtime_error("nll_loss: input must be 2D (N, C)");
    if (target.dtype() != DataType::INT64) throw std::runtime_error("nll_loss: target must be INT64");
    if (target.shape().size() != 1 || target.shape(0) != input.shape(0))
        throw std::runtime_error("nll_loss: target must be 1D of length N");
    Tensor sa, st;
    const Tensor& A = linearize(input, sa);
    const Tensor& T = linearize(target, st);
    require_cpu(A, "nll_loss");
    return loss_result(NllLossImpl<Device::CPU>::execute(A, T), input);
}
Tensor CrossEntropyLoss(const Tensor& input, const Tensor& target) {
    if (input.shape().size() != 2) throw std::runtime_error("cross_entropy: input must be 2D (N, C)");
    Tensor ls = LogSoftmax(input, 1);
    return NllLoss(ls, target);
}

void Typecast(Tensor& a, DataType dst_type) {
    require_contiguous(a, "typecast");
    require_live(a);
    if (!is_cast_valid(a.dtype(), dst_type))
        throw std::runtime_error(std::format("Invalid typecast from {} to {}", dtype_to_string(a.dtype()), dtype_to_string(dst_type)));
    if (a.dtype() == dst_type) return;
    if (is_empty(a)) {
        // 没有元素需要转换：只有存储的元素类型变化。
        a.set_impl(create_tensor_impl(0, dst_type, a.device_id()), dst_type);
        return;
    }
    execute_on<TypecastImpl>(a.device_id(), a, dst_type);
}
Tensor Typecast(const Tensor& a, DataType dst_type) {
    require_live(a);
    if (!is_cast_valid(a.dtype(), dst_type))
        throw std::runtime_error(std::format("Invalid typecast from {} to {}", dtype_to_string(a.dtype()), dtype_to_string(dst_type)));
    if (a.dtype() == dst_type) return a;
    if (is_empty(a)) return empty_on(a, a.shape(), dst_type);
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<TypecastImpl>(A.device_id(), A, dst_type);
}

Tensor Concat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::invalid_argument("tensors is empty");
    if (tensors.size() == 1) return tensors[0];
    DataType dtype = tensors[0].dtype();
    const auto& first_shape = tensors[0].shape();
    const DeviceId first_device = tensors[0].device_id();
    const int ndim = static_cast<int>(first_shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::invalid_argument(std::format("Concat: dim {} is out of range for a {}D tensor", dim, ndim));
    for (const auto& t : tensors) {
        if (t.dtype() != dtype) throw std::invalid_argument("tensors must have same dtype");
        // 只比种类不够：两个 CUDA 实例会让内核用这张卡的 context 去读另一张卡的
        // 指针。
        if (t.device_id() != first_device)
            throw std::invalid_argument(std::format(
                "Concat: all tensors must live on the same device ({} vs {})",
                device_id_to_string(first_device), device_id_to_string(t.device_id())));
        if (t.shape().size() != first_shape.size()) throw std::invalid_argument("tensors must have same number of dimensions");
        for (int i = 0; i < static_cast<int>(first_shape.size()); ++i) {
            if (i != dim && t.shape()[i] != first_shape[i])
                throw std::invalid_argument("tensors must have same shape except concatenation dimension");
        }
    }
    std::vector<int64_t> out_shape(first_shape.begin(), first_shape.end());
    out_shape[dim] = 0;
    for (const auto& t : tensors) out_shape[dim] += t.shape(dim);
    // 只要结果有元素，空输入是允许的；结果本身为空时在这里直接构造，免得内核看到
    // 零工作量。
    if (numel_of(out_shape) == 0) return empty_on(tensors[0], out_shape, dtype);
    // ConcatImpl 线性读各输入；非连续视图先物化（storage 保活），Vulkan 上
    // 带 offset 的视图也要物化，否则内核从 buffer 基址读到视图之外。
    std::vector<Tensor> storage;
    const std::vector<Tensor>* inputs = &tensors;
    for (const auto& t : tensors) {
        if (!t.is_contiguous() || needs_offset_free_input(t)) {
            storage.reserve(tensors.size());
            for (const auto& tt : tensors) storage.push_back(tt.clone());
            inputs = &storage;
            break;
        }
    }
    if (inputs->empty()) throw std::runtime_error("Concat: input list is empty");
    return execute_on<ConcatImpl>((*inputs)[0].device_id(), *inputs, dim);
}

void Transpose(Tensor& a) {
    require_contiguous(a, "transpose");
    require_live(a);
    if (a.shape().size() != 2) throw std::runtime_error("transpose only support 2D tensor");
    if (is_empty(a)) return;
    execute_on<TransposeImpl>(a.device_id(), a);
}
void Transpose(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes) {
    if (a.shape().empty()) throw std::runtime_error("Input tensor must be not null.");
    if (a.shape().size() != axes.size()) throw std::runtime_error("transpose shape and axes size not match");
    if (a.numel() != dst.numel()) throw std::runtime_error("dst numel not enough");
    if (is_empty(a)) return;
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    execute_on<TransposeImpl>(A.device_id(), A, dst, axes);
}
Tensor Transpose(const Tensor& a, std::initializer_list<int64_t> axes) {
    if (a.shape().empty()) throw std::runtime_error("Input tensor must be not null.");
    if (a.shape().size() != axes.size()) throw std::runtime_error("transpose shape and axes size not match");
    if (is_empty(a)) {
        std::vector<int64_t> out_shape;
        out_shape.reserve(axes.size());
        for (int64_t ax : axes) out_shape.push_back(a.shape(ax < 0 ? ax + a.dims() : ax));
        return empty_on(a, out_shape, a.dtype());
    }
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<TransposeImpl>(A.device_id(), A, axes);
}

Tensor Equal(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "equal");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<EqualImpl>(a.device_id(), a, b));
}
Tensor NotEqual(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "not_equal");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<NotEqualImpl>(a.device_id(), a, b));
}
Tensor Greater(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "greater");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<GreaterImpl>(a.device_id(), a, b));
}
Tensor Less(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "less");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<LessImpl>(a.device_id(), a, b));
}
Tensor GreaterEqual(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "greater_equal");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<GreaterEqualImpl>(a.device_id(), a, b));
}
Tensor LessEqual(const Tensor& a, const Tensor& b) {
    if (a.device_id() != b.device_id()) throw std::runtime_error("Tensors a and b device mismatch!");
    if (is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(a, b, "less_equal");
        return empty_on(a, out, DataType::BOOL);
    }

    // 形状由 ufunc 框架的广播规则校验
    return as_bool(execute_on<LessEqualImpl>(a.device_id(), a, b));
}

Tensor Equal(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<EqualImpl>(a.device_id(), a, val));
}
Tensor NotEqual(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<NotEqualImpl>(a.device_id(), a, val));
}
Tensor Greater(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<GreaterImpl>(a.device_id(), a, val));
}
Tensor Less(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<LessImpl>(a.device_id(), a, val));
}
Tensor GreaterEqual(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<GreaterEqualImpl>(a.device_id(), a, val));
}
Tensor LessEqual(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return empty_on(a, a.shape(), DataType::BOOL);
    return as_bool(execute_on<LessEqualImpl>(a.device_id(), a, val));
}

Tensor Where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    if (cond.device_id() != a.device_id() || cond.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    if (is_empty(cond) || is_empty(a) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(cond, a, b, "where");
        return empty_on(cond, out, compute_type(a.dtype(), b.dtype()));
    }
    return execute_on<WhereImpl>(cond.device_id(), cond, a, b);
}
Tensor Where(const Tensor& cond, const Tensor& a, float b) {
    if (cond.device_id() != a.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    if (is_empty(cond) || is_empty(a)) {
        const DimArray out = broadcast_shape_or_throw(cond, a, "where");
        return empty_on(cond, out, scalar_promote_type(a.dtype()));
    }
    return execute_on<WhereImpl>(cond.device_id(), cond, a, b);
}
Tensor Where(const Tensor& cond, float a, const Tensor& b) {
    if (cond.device_id() != b.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    if (is_empty(cond) || is_empty(b)) {
        const DimArray out = broadcast_shape_or_throw(cond, b, "where");
        return empty_on(cond, out, scalar_promote_type(b.dtype()));
    }
    return execute_on<WhereImpl>(cond.device_id(), cond, a, b);
}

bool All(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return true;  // 空真，与 PyTorch 一致
    if (is_bool_dtype(a.dtype())) return All(Typecast(a, DataType::INT8), val);
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<AllImpl>(A.device_id(), A, val);
}
bool Any(const Tensor& a, float val) {
    require_live(a);
    if (is_empty(a)) return false;
    if (is_bool_dtype(a.dtype())) return Any(Typecast(a, DataType::INT8), val);
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<AnyImpl>(A.device_id(), A, val);
}
size_t CountNonzero(const Tensor& a) {
    require_live(a);
    if (is_empty(a)) return 0;
    if (is_bool_dtype(a.dtype())) return CountNonzero(Typecast(a, DataType::INT8));
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<NonZeroImpl>(A.device_id(), A);
}

// 对全部元素做 all/any，返回输入设备上的 0-dim BOOL 张量。真值判定是“非零即真”
// （Python 语义），所以 BOOL 与整型输入先规范化为 0/1 INT8 谓词，再用 min/max 折叠
// ——归约内核在设备侧算出结果，item<bool>() 仍是唯一发生宿主同步的地方。
Tensor All(const Tensor& a) {
    require_live(a);
    const DeviceId id = a.device_id();
    if (is_empty(a)) return execute_on<ops::FillImpl>(id, std::vector<int64_t>{}, DataType::BOOL, 1.0f);
    const Tensor mask = NotEqual(a, 0.0f);    // 输入设备上的 BOOL 张量
    Tensor folded = Min(Typecast(mask, DataType::INT8));
    if (folded.device_id() != id) folded.to_device(id);
    const Tensor& result = folded;
    return Typecast(result, DataType::BOOL);
}
Tensor Any(const Tensor& a) {
    require_live(a);
    const DeviceId id = a.device_id();
    if (is_empty(a)) return execute_on<ops::FillImpl>(id, std::vector<int64_t>{}, DataType::BOOL, 0.0f);
    const Tensor mask = NotEqual(a, 0.0f);
    Tensor folded = Max(Typecast(mask, DataType::INT8));
    if (folded.device_id() != id) folded.to_device(id);
    const Tensor& result = folded;
    return Typecast(result, DataType::BOOL);
}

// 非零元素下标的行主序列表，与 PyTorch 的 nonzero 一致：形状 (count, dims) INT64，
// 1-D 输入为 (count)。仅 CPU：索引内核是宿主循环，设备调用方无论如何都要付一次
// 传输。
Tensor Nonzero(const Tensor& a) {
    require_live(a);
    const size_t dims = a.dims();
    if (is_empty(a)) {
        std::vector<int64_t> shape = {0, static_cast<int64_t>(dims)};
        if (dims == 1) shape = {0};
        return Tensor(std::move(shape), DataType::INT64);
    }
    Tensor host = a.clone();
    host.to_device(DeviceId{Device::CPU, 0});
    if (host.dtype() == DataType::BOOL) {
        const Tensor& chost = host;
        host = ops::Typecast(chost, DataType::INT8);
    }

    // 按调用方 dtype 读扁平数据，收集每个非零元素的坐标。视图先线性化，原因与内核
    // 一致：data() 沿 numel() 个连续元素遍历。
    Tensor flat = host.is_contiguous() ? host : host.contiguous();
    flat = flat.reshape({static_cast<int64_t>(flat.numel())});
    std::vector<int64_t> coords;
    const std::vector<int64_t> shape(a.shape().begin(), a.shape().end());
    const auto visit = [&](auto tag) {
        using T = decltype(tag);
        const T* p = static_cast<const T*>(flat.data());
        const size_t n = flat.numel();
        for (size_t i = 0; i < n; ++i) {
            if (static_cast<double>(p[i]) == 0.0) continue;
            size_t rem = i;
            for (size_t d = dims; d-- > 0;) {
                const size_t s = static_cast<size_t>(shape[d]);
                coords.push_back(static_cast<int64_t>(s ? rem % s : 0));
                rem = s ? rem / s : 0;
                // 上面的压栈顺序是末轴优先；下面反转回来。
            }
            std::reverse(coords.end() - static_cast<std::ptrdiff_t>(dims), coords.end());
        }
    };
    switch (flat.dtype()) {
    case DataType::FLOAT32: visit(float{}); break;
    case DataType::FLOAT64: visit(double{}); break;
    case DataType::FLOAT16: visit(float16{}); break;
    case DataType::BFLOAT16: visit(bfloat16{}); break;
    case DataType::INT8: visit(int8_t{}); break;
    case DataType::INT16: visit(int16_t{}); break;
    case DataType::INT32: visit(int32_t{}); break;
    case DataType::INT64: visit(int64_t{}); break;
    case DataType::BOOL: visit(int8_t{}); break;
    default: throw std::runtime_error("nonzero: unsupported dtype");
    }
    const int64_t count = static_cast<int64_t>(dims ? coords.size() / dims : coords.size());
    std::vector<int64_t> out_shape = {count, static_cast<int64_t>(dims)};
    if (dims == 1) out_shape = {count};
    Tensor out(std::move(out_shape), DataType::INT64);
    if (!coords.empty()) std::memcpy(out.data(), coords.data(), coords.size() * sizeof(int64_t));
    return out;
}
namespace {

// 掩码选择是宿主侧的 gather：两个参数都展平成连续 buffer，单个下标就能按同一
// 行主序遍历它们。mask 的真值按一个字节读取（BOOL 与所有整型都归结为“非零”），
// 因此选择循环不需要按 dtype 派发。
bool mask_byte(const Tensor& m, size_t i) { return static_cast<const int8_t*>(m.data())[i] != 0; }

// 把 `v` 写进连续 CPU 张量的一个元素，并转换为该张量的 dtype。BOOL 规范化为 0/1。
void write_cpu_element(Tensor& flat, size_t i, double v) {
    switch (flat.dtype()) {
    case DataType::FLOAT32: static_cast<float*>(flat.data())[i] = static_cast<float>(v); break;
    case DataType::FLOAT64: static_cast<double*>(flat.data())[i] = v; break;
    case DataType::FLOAT16: static_cast<float16*>(flat.data())[i] = static_cast<float16>(static_cast<float>(v)); break;
    case DataType::BFLOAT16: static_cast<bfloat16*>(flat.data())[i] = static_cast<bfloat16>(static_cast<float>(v)); break;
    case DataType::INT8: static_cast<int8_t*>(flat.data())[i] = static_cast<int8_t>(v); break;
    case DataType::INT16: static_cast<int16_t*>(flat.data())[i] = static_cast<int16_t>(v); break;
    case DataType::INT32: static_cast<int32_t*>(flat.data())[i] = static_cast<int32_t>(v); break;
    case DataType::INT64: static_cast<int64_t*>(flat.data())[i] = static_cast<int64_t>(v); break;
    case DataType::BOOL: static_cast<int8_t*>(flat.data())[i] = v != 0.0 ? 1 : 0; break;
    default: throw std::runtime_error("masked_fill: unsupported dtype");
    }
}

// 把 mask 变成扁平的连续 INT8 buffer，并校验它能广播到 `input`（会放大结果的
// mask 是错误，不是静默广播）。
Tensor flatten_mask(const Tensor& input, const Tensor& mask, std::string_view op) {
    if (!is_bool_dtype(mask.dtype()) && !is_integral_dtype(mask.dtype()))
        throw std::runtime_error(std::format("{}: mask must be BOOL or an integral dtype", op));
    const DimArray bshape = broadcast_shape_or_throw(input, mask, op);
    if (!ops::ufunc::same_shape(std::span<const int64_t>(bshape), input.shape()))
        throw std::runtime_error(std::format(
            "{}: mask must broadcast to the input's shape, not a larger one", op));
    Tensor m = mask.dtype() == DataType::INT8 ? mask : Typecast(mask, DataType::INT8);
    if (!ops::ufunc::same_shape(m.shape(), input.shape()))
        m = m.expand(std::vector<int64_t>(input.shape().begin(), input.shape().end()));
    if (!m.is_contiguous()) m = m.contiguous();
    return m;
}

}  // namespace

Tensor MaskedSelect(const Tensor& input, const Tensor& mask) {
    require_live(input);
    require_live(mask);
    if (input.device_id() != mask.device_id())
        throw std::runtime_error("masked_select: input and mask device mismatch");
    require_cpu(input, "masked_select");
    const Tensor flat_mask = flatten_mask(input, mask, "masked_select");
    Tensor src = input.is_contiguous() ? input : input.contiguous();

    const size_t n = src.numel();
    const size_t esz = calc_dtype_size(src.dtype());
    std::vector<size_t> picked;
    picked.reserve(n);
    for (size_t i = 0; i < n; ++i)
        if (mask_byte(flat_mask, i)) picked.push_back(i);

    Tensor out({static_cast<int64_t>(picked.size())}, input.dtype());
    if (!picked.empty()) {
        // 被选中的元素位置任意，所以逐元素拷贝；字节已是对应目的 dtype。
        auto* dst = static_cast<char*>(out.data());
        const auto* s = static_cast<const char*>(src.data());
        for (size_t k = 0; k < picked.size(); ++k)
            std::memcpy(dst + k * esz, s + picked[k] * esz, esz);
    }
    return out;
}

Tensor MaskedFill(const Tensor& input, const Tensor& mask, float value) {
    require_live(input);
    require_live(mask);
    if (input.device_id() != mask.device_id())
        throw std::runtime_error("masked_fill: input and mask device mismatch");
    require_cpu(input, "masked_fill");
    const Tensor flat_mask = flatten_mask(input, mask, "masked_fill");
    Tensor out = input.is_contiguous() ? input.clone() : input.contiguous();
    out = out.reshape({static_cast<int64_t>(out.numel())});
    const size_t n = out.numel();
    for (size_t i = 0; i < n; ++i)
        if (mask_byte(flat_mask, i)) write_cpu_element(out, i, value);
    return out.reshape(std::vector<int64_t>(input.shape().begin(), input.shape().end()));
}

// ===== FR-11：形状 / 数据搬运算子 =====
// Split/Chunk/Roll/Flip/Pad/Repeat 全部用视图、Concat 与 Fill 拼成——这些原语每个
// 后端都已实现——因此设备上不需要自己的内核。Sort/Topk 需要内核，只有 CPU 实现；
// 设备调用走 FR-8 的“未实现”路径。

std::vector<Tensor> Split(const Tensor& t, int64_t split_size, int64_t dim) {
    require_live(t);
    if (split_size <= 0)
        throw std::runtime_error(std::format("split: split_size must be positive, got {}", split_size));
    const int d = normalize_axis(static_cast<int>(dim), t.dims(), "split");
    const int64_t size = t.shape(d);
    std::vector<Tensor> parts;
    if (size == 0) {
        // 长度为 0 的轴仍会切分：返回一个空视图，而不是空列表。
        parts.push_back(t.narrow(d, 0, 0));
        return parts;
    }
    for (int64_t start = 0; start < size; start += split_size)
        parts.push_back(t.narrow(d, start, std::min(split_size, size - start)));
    return parts;
}

std::vector<Tensor> Chunk(const Tensor& t, int64_t chunks, int64_t dim) {
    require_live(t);
    if (chunks <= 0)
        throw std::runtime_error(std::format("chunk: chunks must be positive, got {}", chunks));
    const int d = normalize_axis(static_cast<int>(dim), t.dims(), "chunk");
    const int64_t size = t.shape(d);
    const int64_t split_size = (size + chunks - 1) / chunks;
    // 空轴推不出块长；按请求数量返回空视图，调用方仍可照常 zip（PyTorch 也是）。
    if (split_size == 0) return std::vector<Tensor>(static_cast<size_t>(chunks), t);
    return Split(t, split_size, d);
}

Tensor Stack(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty())
        throw std::invalid_argument("stack: the tensor list is empty");
    const Tensor& first = tensors[0];
    require_live(first);
    const int ndim = static_cast<int>(first.dims()) + 1;
    int d = static_cast<int>(dim);
    if (d < 0) d += ndim;
    if (d < 0 || d >= ndim)
        throw std::runtime_error(std::format(
            "stack: dim {} is out of range for a {}D result", dim, ndim));
    std::vector<Tensor> raised;
    raised.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.dtype() != first.dtype() || t.device_id() != first.device_id())
            throw std::runtime_error("stack: all tensors must have the same dtype and device");
        if (!ops::ufunc::same_shape(t.shape(), first.shape()))
            throw std::runtime_error("stack: all tensors must have the same shape");
        raised.push_back(t.unsqueeze(d));
    }
    return Concat(raised, d);
}

Tensor Pad(const Tensor& t, const std::vector<int64_t>& pads, float value) {
    require_live(t);
    if (pads.size() % 2 != 0)
        throw std::runtime_error("pad: pads must be a flat list of (begin, end) pairs");
    const size_t npairs = pads.size() / 2;
    if (npairs > t.dims())
        throw std::runtime_error(std::format(
            "pad: {} pad pairs for a {}D tensor", npairs, t.dims()));
    // 从最后一维向前处理，与 F.pad 的布局完全一致：先裁剪（负数填充），再用 Concat
    // 在两侧粘上常量块。
    Tensor cur = t;
    for (size_t i = 0; i < npairs; ++i) {
        const int d = static_cast<int>(t.dims() - 1 - i);
        const int64_t begin_pad = std::max<int64_t>(pads[2 * i], 0);
        const int64_t end_pad = std::max<int64_t>(pads[2 * i + 1], 0);
        const int64_t size = cur.shape(d);
        const int64_t start = std::min(std::max<int64_t>(-pads[2 * i], 0), size);
        const int64_t stop = std::max<int64_t>(size - std::max<int64_t>(-pads[2 * i + 1], 0), start);
        cur = cur.narrow(d, start, stop - start);
        if (begin_pad == 0 && end_pad == 0) continue;

        std::vector<int64_t> block_shape(cur.shape().begin(), cur.shape().end());
        std::vector<Tensor> parts;
        if (begin_pad > 0) {
            block_shape[static_cast<size_t>(d)] = begin_pad;
            parts.push_back(filled_on(cur, block_shape, cur.dtype(), value));
        }
        parts.push_back(cur);
        if (end_pad > 0) {
            block_shape[static_cast<size_t>(d)] = end_pad;
            parts.push_back(filled_on(cur, block_shape, cur.dtype(), value));
        }
        cur = Concat(parts, d);
    }
    return cur;
}

Tensor Repeat(const Tensor& t, const std::vector<int64_t>& repeats) {
    require_live(t);
    if (repeats.empty())
        throw std::runtime_error("repeat: repeats must not be empty");
    for (int64_t r : repeats)
        if (r < 0)
            throw std::runtime_error(std::format("repeat: negative repeat count {}", r));
    const size_t n = repeats.size();
    if (t.dims() > n)
        throw std::runtime_error(std::format(
            "repeat: {} repeat entries for a {}D tensor", n, t.dims()));

    // NumPy tile：把操作数右对齐，每轴拆成 (1, size)，把 1 广播到重复次数，
    // 再把每一对折回一起。
    std::vector<int64_t> interleaved, expanded, out_shape(n);
    interleaved.reserve(2 * n);
    expanded.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        const int64_t size = i + t.dims() >= n ? t.shape(i - (n - t.dims())) : 1;
        interleaved.push_back(1);
        interleaved.push_back(size);
        expanded.push_back(repeats[i]);
        expanded.push_back(size);
        out_shape[i] = repeats[i] * size;
    }
    Tensor r = t.reshape(interleaved);
    return r.expand(expanded).reshape(out_shape);
}

Tensor Flip(const Tensor& t, const std::vector<int64_t>& dims) {
    require_live(t);
    Tensor cur = t;
    for (int64_t raw : dims) {
        const int d = normalize_axis(static_cast<int>(raw), cur.dims(), "flip");
        const int64_t size = cur.shape(d);
        if (size <= 1) continue;
        // 逆序取视图；Concat 一遍读完。
        std::vector<Tensor> parts;
        parts.reserve(static_cast<size_t>(size));
        for (int64_t i = size - 1; i >= 0; --i) parts.push_back(cur.narrow(d, i, 1));
        cur = Concat(parts, d);
    }
    return cur;
}

Tensor Roll(const Tensor& t, int64_t shift, int64_t dim) {
    require_live(t);
    const int d = normalize_axis(static_cast<int>(dim), t.dims(), "roll");
    const int64_t size = t.shape(d);
    if (size == 0) return t;
    int64_t s = shift % size;
    if (s < 0) s += size;
    if (s == 0) return t;
    std::vector<Tensor> parts{t.narrow(d, size - s, s), t.narrow(d, 0, size - s)};
    return Concat(parts, d);
}

SortResult Sort(const Tensor& t, int64_t dim, bool descending) {
    require_live(t);
    require_not_bool(t, "sort");
    const int d = normalize_axis(static_cast<int>(dim), t.dims(), "sort");
    Tensor storage;
    const Tensor& A = linearize(t, storage);
    return execute_on<SortImpl>(A.device_id(), A, d, descending);
}

TopkResult Topk(const Tensor& t, int64_t k, int64_t dim, bool largest) {
    require_live(t);
    require_not_bool(t, "topk");
    if (k <= 0) throw std::runtime_error(std::format("topk: k must be positive, got {}", k));
    const int d = normalize_axis(static_cast<int>(dim), t.dims(), "topk");
    Tensor storage;
    const Tensor& A = linearize(t, storage);
    return execute_on<TopkImpl>(A.device_id(), A, k, d, largest);
}

Tensor Argmax(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    axis = normalize_axis(axis, a.dims(), "argmax");
    if (is_bool_dtype(a.dtype())) return Argmax(Typecast(a, DataType::INT8), axis, keepdim);
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0)
        throw std::runtime_error("argmax: an empty axis has no index");
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, DataType::INT64);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    Tensor result = execute_on<ArgMaxImpl>(A.device_id(), A, axis);
    // 即使后端原生索引内核是 INT32，契约输出仍是 INT64。
    if (result.dtype() != DataType::INT64) result = result.to_type(DataType::INT64);
    return with_reduced_shape(std::move(result), a.shape(), axis, keepdim);
}
Tensor Argmin(const Tensor& a, int axis, bool keepdim) {
    require_live(a);
    axis = normalize_axis(axis, a.dims(), "argmin");
    if (is_bool_dtype(a.dtype())) return Argmin(Typecast(a, DataType::INT8), axis, keepdim);
    const std::vector<int64_t> out_shape = reduced_shape(a.shape(), axis, keepdim);
    if (a.shape(axis) == 0)
        throw std::runtime_error("argmin: an empty axis has no index");
    if (numel_of(out_shape) == 0) return empty_on(a, out_shape, DataType::INT64);
    Tensor storage;
    const Tensor& A = prepare_reduce_input(a, storage, axis);
    Tensor result = execute_on<ArgMinImpl>(A.device_id(), A, axis);
    if (result.dtype() != DataType::INT64) result = result.to_type(DataType::INT64);
    return with_reduced_shape(std::move(result), a.shape(), axis, keepdim);
}

Tensor IndexSelect(const Tensor& input, int dim, const Tensor& index) {
    if (input.device_id() != index.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    if (index.dtype() != DataType::INT32 && index.dtype() != DataType::INT64)
        throw std::runtime_error("index_select: index must be INT32 or INT64");
    if (index.shape().size() != 1)
        throw std::runtime_error("index_select: index must be 1D");
    dim = dim < 0 ? dim + static_cast<int>(input.shape().size()) : dim;
    if (dim < 0 || dim >= static_cast<int>(input.shape().size()))
        throw std::runtime_error("index_select: dim out of range");
    if (index.numel() == 0) {
        std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
        out_shape[dim] = 0;
        return empty_on(input, out_shape, input.dtype());
    }
    return execute_on<IndexSelectImpl>(input.device_id(), input, dim, index);
}

Tensor Gather(const Tensor& input, int dim, const Tensor& index) {
    if (input.device_id() != index.device_id())
        throw std::runtime_error("Tensor Device mismatch!");
    if (index.dtype() != DataType::INT32 && index.dtype() != DataType::INT64)
        throw std::runtime_error("gather: index must be INT32 or INT64");
    dim = dim < 0 ? dim + static_cast<int>(input.shape().size()) : dim;
    if (dim < 0 || dim >= static_cast<int>(input.shape().size()))
        throw std::runtime_error("gather: dim out of range");
    // 秩与非 dim 维的尺寸都在这里校验：设备内核直接用 index 的形状解码输入坐标，
    // 放一个更宽的 index 过去就是越界读。
    if (index.shape().size() != input.shape().size())
        throw std::runtime_error("gather: index must have the same rank as input");
    for (size_t d = 0; d < index.shape().size(); ++d)
        if (static_cast<int>(d) != dim && index.shape(d) > input.shape(d))
            throw std::runtime_error("gather: index size exceeds input along non-dim axes");
    if (index.numel() == 0) return empty_on(input, index.shape(), input.dtype());
    return execute_on<GatherImpl>(input.device_id(), input, dim, index);
}

namespace {

// 所有已启动的后端，这样一次调用就能覆盖本程序实际用过的设备，无需调用方逐个
// 点名。
void for_each_live_control(std::function<void()> DeviceControl::*fn) {
    for (const auto& entry : live_device_controls()) {
        const std::function<void()>& callback = entry.second.*fn;
        if (callback) callback();
    }
}

}  // namespace

void synchronize() {
    for_each_live_control(&DeviceControl::synchronize);
}

void empty_cache() {
    for_each_live_control(&DeviceControl::empty_cache);
}

int device_count(Device kind) {
    return device_probe(kind).count;
}

int current_device(Device kind) {
    return current_device_index(kind);
}

void set_device(Device kind, int index) {
    set_current_device_index(kind, index);
}

DeviceGuard::DeviceGuard(DeviceId id)
    : m_previous{id.kind, current_device_index(id.kind)} {
    // 常见情况是算子就落在已是 current 的设备上；跳过 set 让每算子守卫只剩一次
    // 线程私有查找。
    if (m_previous.index != id.index) set_current_device_index(id.kind, id.index);
}

DeviceGuard::~DeviceGuard() {
    // 析构函数不抛异常。构造守卫时索引是有效的，所以只有在设备集合被缩小时恢复才会
    // 失败——本库从不这样做。
    try {
        if (current_device_index(m_previous.kind) != m_previous.index)
            set_current_device_index(m_previous.kind, m_previous.index);
    } catch (...) {
    }
}

}  // namespace ops
