#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "interface/tensorvia.hpp"
#include "interface/types.hpp"
#include "interface/tensor.hpp"
#include "interface/ops.hpp"
#include "interface/cpu_ops.hpp"
#include "interface/backend/vulkan/vulkan_ops.hpp"  // declarations only; definitions are compiled under BACKEND_VULKAN
#include "interface/backend/cuda/cuda_ops.hpp"    // declarations only; definitions are compiled under BACKEND_CUDA
#include "interface/backend/sycl/sycl_ops.hpp"    // declarations only; definitions are compiled under BACKEND_SYCL

namespace {

// Route an op to the backend that owns the tensor(s). Only backends compiled
// into this artifact are listed; asking for anything else is an error, never a
// silent fallback (CPU kernels cannot dereference device buffers).
// The primary Impl templates are declared but not defined, so an unsupported
// backend is detectable without instantiating execute(). This matters because
// :vulkan_ops / :cuda_ops only declare the operators their backend implements.
template <template <Device> class Impl>
concept has_vulkan_impl = requires { sizeof(Impl<Device::VULKAN>); };

template <template <Device> class Impl>
concept has_cuda_impl = requires { sizeof(Impl<Device::CUDA>); };

template <template <Device> class Impl>
concept has_sycl_impl = requires { sizeof(Impl<Device::SYCL>); };

template <template <Device> class Impl, typename... Args>
decltype(auto) execute_on(Device device, Args&&... args) {
    switch (device) {
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
            "no implementation compiled for device {}", device_to_string(device)));
    }
}

inline void require_same_device(std::string_view op, std::initializer_list<Device> devices) {
    const Device first = *devices.begin();
    for (Device d : devices) {
        if (d != first)
            throw std::runtime_error(std::format("{}: tensors must live on the same device", op));
    }
}

inline void require_cpu(const Tensor& t, std::string_view op) {
    if (t.device() != Device::CPU)
        throw std::runtime_error(std::format(
            "{}: only the CPU backend implements this op (tensor is on {})",
            op, device_to_string(t.device())));
}

// slice()/t()/permute() 返回共享底层 buffer 的视图，而计算 kernel 只支持从
// Tensor::data()（已含视图 offset）起线性遍历 numel 个元素：
//  - 连续视图（含带 offset 的切片）：线性遍历正好命中视图元素，零成本直通；
//  - 非连续视图（t()/permute()/跨步切片）：先物化成连续拷贝再进 kernel，
//    否则线性遍历会读到视图之外的共享内存。
// 这是 strided 内核接入前的正确性守卫；storage 保活物化结果直到调用结束。
const Tensor& linearize(const Tensor& t, Tensor& storage) {
    if (t.is_contiguous()) return t;
    storage = t.contiguous();
    return storage;
}

// 写入型（in-place / dst 输出）操作无法在非连续视图上线性读写：
// 会破坏视图之外的共享内存，也无法保持"写穿到原张量"的语义 —— 明确拒绝。
void require_contiguous(const Tensor& t, std::string_view op) {
    if (!t.is_contiguous())
        throw std::runtime_error(std::format(
            "{}: non-contiguous view is not supported here; call to_contiguous() first", op));
}

}  // namespace

namespace ops {

void println(Tensor& a) {
    if (a.numel() == 0 || a.data() == nullptr) throw std::runtime_error("tensor a is null");
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
    if (a.numel() == 0 || a.data() == nullptr) throw std::runtime_error("tensor a is null");
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
    if (a.numel() == 0 || a.data() == nullptr) throw std::runtime_error("tensor a is null");
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
    if (a.numel() == 0 || a.data() == nullptr) throw std::runtime_error("tensor a is null");
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
    if (shape.empty()) throw std::runtime_error("ops::Ones: shape is empty");
    return execute_on<OnesImpl>(device, shape, dtype);
}

Tensor Zeros(const std::vector<int64_t>& shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("ops::Zeros: shape is empty");
    return execute_on<ZerosImpl>(device, shape, dtype);
}

Tensor Fill(const std::vector<int64_t>& shape, DataType dtype, float val, Device device) {
    if (shape.empty()) throw std::runtime_error("ops::Fill: shape is empty");
    return execute_on<FillImpl>(device, shape, dtype, val);
}

Tensor Random(const std::vector<int64_t>& shape, DataType dtype, float min, float max, Device device) {
    if (shape.empty()) throw std::runtime_error("ops::Random: shape is empty");
    return execute_on<RandomImpl>(device, shape, dtype, min, max);
}

Tensor Randn(const std::vector<int64_t>& shape, DataType dtype, float mean, float std, Device device) {
    if (shape.empty()) throw std::runtime_error("ops::Randn: shape is empty");
    if (std < 0.0f) throw std::runtime_error("randn: std must be non-negative");
    if (device != Device::CPU)
        throw std::runtime_error("randn: only the CPU backend implements this op");
    return execute_on<RandnImpl>(device, shape, dtype, mean, std);
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
    return execute_on<SliceImpl>(t.device(), t, ranges);
}

void Add(Tensor& a, float b) {
    if (a.numel() == 0 || b == 0.0f) return;
    execute_on<AddImpl>(a.device(), a, b);
}
void Sub(Tensor& a, float b) {
    if (a.numel() == 0 || b == 0.0f) return;
    execute_on<SubImpl>(a.device(), a, b);
}
void Mul(Tensor& a, float b) {
    if (a.numel() == 0 || b == 1.0f) return;
    execute_on<MulImpl>(a.device(), a, b);
}
void Div(Tensor& a, float b) {
    if (a.numel() == 0 || b == 1.0f) return;
    if (b == 0.0f) throw std::runtime_error("div: Cann't division by zero");
    execute_on<DivImpl>(a.device(), a, b);
}

Tensor Add(const Tensor& a, float b) {
    if (a.numel() == 0 || b == 0.0f) return a.clone();
    return execute_on<AddImpl>(a.device(), a, b);
}
Tensor Sub(const Tensor& a, float b) {
    if (a.numel() == 0 || b == 0.0f) return a.clone();
    return execute_on<SubImpl>(a.device(), a, b);
}
Tensor Mul(const Tensor& a, float b) {
    if (a.numel() == 0 || b == 1.0f) return a.clone();
    return execute_on<MulImpl>(a.device(), a, b);
}
Tensor Div(const Tensor& a, float b) {
    if (a.numel() == 0 || b == 1.0f) return a.clone();
    if (b == 0.0f) throw std::runtime_error("Division by zero");
    return execute_on<DivImpl>(a.device(), a, b);
}

Tensor Add(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensor Device mismatch!");

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<AddImpl>(a.device(), a, b);
}
Tensor Sub(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensor Device mismatch!");

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<SubImpl>(a.device(), a, b);
}
Tensor Mul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensor Device mismatch!");

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<MulImpl>(a.device(), a, b);
}
Tensor Div(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Device mismatch!");

    // shape 校验由 ufunc 框架的广播规则处理
    return execute_on<DivImpl>(a.device(), a, b);
}

void Add(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device() != b.device() || dst.device() != a.device() || dst.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<AddImpl>(a.device(), a, b, dst);
}
void Sub(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device() != b.device() || dst.device() != a.device() || dst.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<SubImpl>(a.device(), a, b, dst);
}
void Mul(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device() != b.device() || dst.device() != a.device() || dst.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<MulImpl>(a.device(), a, b, dst);
}
void Div(const Tensor& a, const Tensor& b, Tensor& dst) {
    if (a.device() != b.device() || dst.device() != a.device() || dst.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");

    // dst 形状须等于广播结果，由 ufunc 框架校验
    execute_on<DivImpl>(a.device(), a, b, dst);
}

Tensor MatMul(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device())
        throw std::runtime_error("Tensors a and b device mismatch!");
    if (a.shape().size() != b.shape().size())
        throw std::runtime_error(std::format("Tensors a and b must have the same rank,but got {} and {}", a.shape().size(), b.shape().size()));
    if (a.shape().size() != 2 && a.shape().size() != 3)
        throw std::runtime_error(std::format("Tensors a and b dims must be 2 or 3,but got {}", a.shape().size()));
    if (a.shape().size() == 3) {
        if (a.shape(0) != b.shape(0))
            throw std::runtime_error("Tensors a and b must have the same batch size");
        if (a.shape(2) != b.shape(1))
            throw std::runtime_error(std::format("Tensors common dim must match,but {} and {}", a.shape(2), b.shape(1)));
    } else if (a.shape().size() == 2) {
        if (a.shape(1) != b.shape(0))
            throw std::runtime_error(std::format("Tensors common dim must match,but {} and {}", a.shape(0), b.shape(0)));
    }
    Tensor storage_a, storage_b;
    const Tensor& A = linearize(a, storage_a);
    const Tensor& B = linearize(b, storage_b);
    return execute_on<MatMulImpl>(A.device(), A, B);
}

Tensor Abs(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<AbsImpl>(a.device(), a);
}
Tensor Sin(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<SinImpl>(a.device(), a);
}
Tensor Cos(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<CosImpl>(a.device(), a);
}
Tensor Tan(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<TanImpl>(a.device(), a);
}
Tensor Exp(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<ExpImpl>(a.device(), a);
}
Tensor Relu(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<ReluImpl>(a.device(), a);
}
Tensor Silu(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<SiluImpl>(a.device(), a);
}
Tensor Tanh(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<TanhImpl>(a.device(), a);
}
Tensor Sqrt(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (ops::Any((a < 0), 1)) throw std::runtime_error("tensor must be non-negative");
    return execute_on<SqrtImpl>(a.device(), a);
}
Tensor Sigmoid(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<SigmoidImpl>(a.device(), a);
}
Tensor Pow(const Tensor& a, float val) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    return execute_on<PowImpl>(a.device(), a, val);
}
Tensor Log(const Tensor& a, float val) {
    if (val < 0 || val == 1) throw std::runtime_error("The base cannot be less than or equal to 0, and it cannot be 1");
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (ops::Any((a < 0), 1)) throw std::runtime_error("log: tensor must be non-negative");
    return execute_on<LogImpl>(a.device(), a, val);
}
Tensor Softmax(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (std::abs(axis) >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<SoftmaxImpl>(A.device(), A, axis);
}
Tensor LogSoftmax(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (std::abs(axis) >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "log_softmax");
    return LogSoftmaxImpl<Device::CPU>::execute(A, axis);
}
Tensor Erf(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    require_cpu(a, "erf");
    return ErfImpl<Device::CPU>::execute(a);
}
Tensor LayerNorm(const Tensor& a, const Tensor& weight, const Tensor& bias, int axis, float eps) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (eps <= 0.0f) throw std::runtime_error("layer_norm: eps must be positive");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis < 0 || axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("layer_norm: axis out of range");
    auto a_shape = a.shape();
    int dim = a_shape.size();
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];
    DataType res_type = a.dtype() <= DataType::INT64 ? DataType::FLOAT32 : a.dtype();
    if (weight.numel() != norm_size) throw std::runtime_error("layer_norm: weight numel must match normalized block size");
    if (bias.numel() != norm_size) throw std::runtime_error("layer_norm: bias numel must match normalized block size");
    if (weight.dtype() != res_type || bias.dtype() != res_type)
        throw std::runtime_error("layer_norm: weight/bias dtype must match result dtype");
    Tensor sa, sw, sb;
    const Tensor& A = linearize(a, sa);
    const Tensor& W = linearize(weight, sw);
    const Tensor& B = linearize(bias, sb);
    require_cpu(A, "layer_norm");
    return LayerNormImpl<Device::CPU>::execute(A, W, B, axis, eps);
}
Tensor RmsNorm(const Tensor& a, const Tensor& weight, int axis, float eps) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (eps <= 0.0f) throw std::runtime_error("rms_norm: eps must be positive");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis < 0 || axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("rms_norm: axis out of range");
    auto a_shape = a.shape();
    int dim = static_cast<int>(a_shape.size());
    size_t norm_size = 1;
    for (int i = axis; i < dim; ++i) norm_size *= a_shape[i];
    DataType res_type = a.dtype() <= DataType::INT64 ? DataType::FLOAT32 : a.dtype();
    if (weight.numel() != norm_size) throw std::runtime_error("rms_norm: weight numel must match normalized block size");
    if (weight.dtype() != res_type)
        throw std::runtime_error("rms_norm: weight dtype must match result dtype");
    Tensor sa, sw;
    const Tensor& A = linearize(a, sa);
    const Tensor& W = linearize(weight, sw);
    require_cpu(A, "rms_norm");
    return RmsNormImpl<Device::CPU>::execute(A, W, axis, eps);
}
Tensor Clamp(const Tensor& a, float min, float max) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (min > max) throw std::runtime_error("min must be less than max");
    return execute_on<ClampImpl>(a.device(), a, min, max);
}

void Zeros(Tensor& a) {
    require_contiguous(a, "zeros");
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.device() != Device::CPU) {
        execute_on<FillImpl>(a.device(), a, 0.0f);
        return;
    }
    // 所有支持 dtype（int / float / float16 / bfloat16）的零表示都是全零 bit pattern，
    // memset 比走 dispatch_dtype 的循环快得多。
    std::memset(a.data(), 0, a.numel() * calc_dtype_size(a.dtype()));
}

void Fill(Tensor& a, float value) {
    require_contiguous(a, "fill");
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.device() != Device::CPU) {
        execute_on<FillImpl>(a.device(), a, value);
        return;
    }
    dispatch_dtype(a.dtype(), [&](auto type_id) {
        using T = typename decltype(type_id)::type;
        T* p = static_cast<T*>(a.data());
        std::fill(p, p + a.numel(), static_cast<T>(value));
    });
}

void Ones(Tensor& a) {
    Fill(a, 1.0f);
}

void Abs(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    execute_on<AbsImpl>(a.device(), a);
}
void Sin(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<SinImpl>(a.device(), a);
}
void Cos(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<CosImpl>(a.device(), a);
}
void Tan(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<TanImpl>(a.device(), a);
}
void Relu(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    execute_on<ReluImpl>(a.device(), a);
}
void Silu(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<SiluImpl>(a.device(), a);
}
void Tanh(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<TanhImpl>(a.device(), a);
}
void Pow(Tensor& a, float val) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    execute_on<PowImpl>(a.device(), a, val);
}
void Sigmoid(Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.dtype() < DataType::BFLOAT16)
        throw std::runtime_error(std::format("{} of tensor a is not supported", dtype_to_string(a.dtype())));
    execute_on<SigmoidImpl>(a.device(), a);
}
void Clamp(Tensor& a, float min, float max) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (min > max) throw std::runtime_error("min must be less than max");
    execute_on<ClampImpl>(a.device(), a, min, max);
}

float Sum(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<SumImpl>(A.device(), A);
}
float Min(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MinImpl>(A.device(), A);
}
float Max(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MaxImpl>(A.device(), A);
}
float Mean(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MeanImpl>(A.device(), A);
}
float Prod(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "prod");
    return ProdImpl<Device::CPU>::execute(A);
}
float Var(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "var");
    return VarImpl<Device::CPU>::execute(A);
}
float Std(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "std");
    return StdImpl<Device::CPU>::execute(A);
}
float Norm(const Tensor& a, float p) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (p != 1.0f && p != 2.0f) throw std::runtime_error("norm: only p=1 or p=2 supported");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "norm");
    return NormImpl<Device::CPU>::execute(A, p);
}

Tensor Sum(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<SumImpl>(A.device(), A, axis);
}
Tensor Min(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MinImpl>(A.device(), A, axis);
}
Tensor Max(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MaxImpl>(A.device(), A, axis);
}
Tensor Mean(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<MeanImpl>(A.device(), A, axis);
}
Tensor Prod(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "prod");
    return ProdImpl<Device::CPU>::execute(A, axis);
}
Tensor Var(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "var");
    return VarImpl<Device::CPU>::execute(A, axis);
}
Tensor Std(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "std");
    return StdImpl<Device::CPU>::execute(A, axis);
}
Tensor Norm(const Tensor& a, float p, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (p != 1.0f && p != 2.0f) throw std::runtime_error("norm: only p=1 or p=2 supported");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "norm");
    return NormImpl<Device::CPU>::execute(A, p, axis);
}
Tensor Cumsum(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    require_cpu(A, "cumsum");
    return CumsumImpl<Device::CPU>::execute(A, axis);
}

float MseLoss(const Tensor& input, const Tensor& target) {
    if (input.device() != target.device()) throw std::runtime_error("Tensor Device mismatch!");
    if (input.numel() == 0) throw std::runtime_error("mse_loss: empty input");
    if (input.dtype() != target.dtype()) throw std::runtime_error("mse_loss: input and target must have same dtype");
    if (input.shape().size() != target.shape().size() || input.numel() != target.numel())
        throw std::runtime_error("mse_loss: input and target must have same shape");
    Tensor sa, sb;
    const Tensor& A = linearize(input, sa);
    const Tensor& B = linearize(target, sb);
    require_cpu(A, "mse_loss");
    return MseLossImpl<Device::CPU>::execute(A, B);
}
float NllLoss(const Tensor& input, const Tensor& target) {
    if (input.device() != target.device()) throw std::runtime_error("Tensor Device mismatch!");
    if (input.numel() == 0) throw std::runtime_error("nll_loss: empty input");
    if (input.shape().size() != 2) throw std::runtime_error("nll_loss: input must be 2D (N, C)");
    if (target.dtype() != DataType::INT64) throw std::runtime_error("nll_loss: target must be INT64");
    if (target.shape().size() != 1 || target.shape(0) != input.shape(0))
        throw std::runtime_error("nll_loss: target must be 1D of length N");
    Tensor sa, st;
    const Tensor& A = linearize(input, sa);
    const Tensor& T = linearize(target, st);
    require_cpu(A, "nll_loss");
    return NllLossImpl<Device::CPU>::execute(A, T);
}
float CrossEntropyLoss(const Tensor& input, const Tensor& target) {
    if (input.shape().size() != 2) throw std::runtime_error("cross_entropy: input must be 2D (N, C)");
    Tensor ls = LogSoftmax(input, 1);
    return NllLoss(ls, target);
}

void Typecast(Tensor& a, DataType dst_type) {
    require_contiguous(a, "typecast");
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (!is_cast_valid(a.dtype(), dst_type))
        throw std::runtime_error(std::format("Invalid typecast from {} to {}", dtype_to_string(a.dtype()), dtype_to_string(dst_type)));
    if (a.dtype() == dst_type) return;
    execute_on<TypecastImpl>(a.device(), a, dst_type);
}
Tensor Typecast(const Tensor& a, DataType dst_type) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (!is_cast_valid(a.dtype(), dst_type))
        throw std::runtime_error(std::format("Invalid typecast from {} to {}", dtype_to_string(a.dtype()), dtype_to_string(dst_type)));
    if (a.dtype() == dst_type) return a;
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<TypecastImpl>(A.device(), A, dst_type);
}

Tensor Concat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) throw std::invalid_argument("tensors is empty");
    if (tensors.size() == 1) return tensors[0];
    DataType dtype = tensors[0].dtype();
    const auto& first_shape = tensors[0].shape();
    for (const auto& t : tensors) {
        if (t.dtype() != dtype) throw std::invalid_argument("tensors must have same dtype");
        if (t.shape().size() != first_shape.size()) throw std::invalid_argument("tensors must have same number of dimensions");
        for (int i = 0; i < static_cast<int>(first_shape.size()); ++i) {
            if (i != dim && t.shape()[i] != first_shape[i])
                throw std::invalid_argument("tensors must have same shape except concatenation dimension");
        }
    }
    // ConcatImpl 线性读各输入的 data()；非连续输入先物化（storage 保活）。
    std::vector<Tensor> storage;
    const std::vector<Tensor>* inputs = &tensors;
    for (const auto& t : tensors) {
        if (!t.is_contiguous()) {
            storage.reserve(tensors.size());
            for (const auto& tt : tensors) storage.push_back(tt.contiguous());
            inputs = &storage;
            break;
        }
    }
    if (inputs->empty()) throw std::runtime_error("Concat: input list is empty");
    return execute_on<ConcatImpl>((*inputs)[0].device(), *inputs, dim);
}

void Transpose(Tensor& a) {
    require_contiguous(a, "transpose");
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    if (a.shape().size() != 2) throw std::runtime_error("transpose only support 2D tensor");
    execute_on<TransposeImpl>(a.device(), a);
}
void Transpose(const Tensor& a, Tensor& dst, std::initializer_list<int64_t> axes) {
    if (a.shape().empty()) throw std::runtime_error("Input tensor must be not null.");
    if (a.shape().size() != axes.size()) throw std::runtime_error("transpose shape and axes size not match");
    if (a.numel() != dst.numel()) throw std::runtime_error("dst numel not enough");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    execute_on<TransposeImpl>(A.device(), A, dst, axes);
}
Tensor Transpose(const Tensor& a, std::initializer_list<int64_t> axes) {
    if (a.shape().empty()) throw std::runtime_error("Input tensor must be not null.");
    if (a.shape().size() != axes.size()) throw std::runtime_error("transpose shape and axes size not match");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<TransposeImpl>(A.device(), A, axes);
}

Tensor Equal(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<EqualImpl>(a.device(), a, b);
}
Tensor NotEqual(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<NotEqualImpl>(a.device(), a, b);
}
Tensor Greater(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<GreaterImpl>(a.device(), a, b);
}
Tensor Less(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<LessImpl>(a.device(), a, b);
}
Tensor GreaterEqual(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<GreaterEqualImpl>(a.device(), a, b);
}
Tensor LessEqual(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) throw std::runtime_error("Tensors a and b device mismatch!");

    // 形状由 ufunc 框架的广播规则校验
    return execute_on<LessEqualImpl>(a.device(), a, b);
}

Tensor Equal(const Tensor& a, float val) { return execute_on<EqualImpl>(a.device(), a, val); }
Tensor NotEqual(const Tensor& a, float val) { return execute_on<NotEqualImpl>(a.device(), a, val); }
Tensor Greater(const Tensor& a, float val) { return execute_on<GreaterImpl>(a.device(), a, val); }
Tensor Less(const Tensor& a, float val) { return execute_on<LessImpl>(a.device(), a, val); }
Tensor GreaterEqual(const Tensor& a, float val) { return execute_on<GreaterEqualImpl>(a.device(), a, val); }
Tensor LessEqual(const Tensor& a, float val) { return execute_on<LessEqualImpl>(a.device(), a, val); }

Tensor Where(const Tensor& cond, const Tensor& a, const Tensor& b) {
    if (cond.device() != a.device() || cond.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");
    require_cpu(cond, "where");
    return WhereImpl<Device::CPU>::execute(cond, a, b);
}
Tensor Where(const Tensor& cond, const Tensor& a, float b) {
    if (cond.device() != a.device())
        throw std::runtime_error("Tensor Device mismatch!");
    require_cpu(cond, "where");
    return WhereImpl<Device::CPU>::execute(cond, a, b);
}
Tensor Where(const Tensor& cond, float a, const Tensor& b) {
    if (cond.device() != b.device())
        throw std::runtime_error("Tensor Device mismatch!");
    require_cpu(cond, "where");
    return WhereImpl<Device::CPU>::execute(cond, a, b);
}

bool All(const Tensor& a, float val) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<AllImpl>(A.device(), A, val);
}
bool Any(const Tensor& a, float val) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<AnyImpl>(A.device(), A, val);
}
size_t Nonzero(const Tensor& a) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<NonZeroImpl>(A.device(), A);
}
Tensor Argmax(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<ArgMaxImpl>(A.device(), A, axis);
}
Tensor Argmin(const Tensor& a, int axis) {
    if (a.data() == nullptr || a.numel() == 0) throw std::runtime_error("tensor a is null");
    axis = axis < 0 ? axis + static_cast<int>(a.shape().size()) : axis;
    if (axis >= static_cast<int>(a.shape().size())) throw std::runtime_error("axis is out of range");
    Tensor storage;
    const Tensor& A = linearize(a, storage);
    return execute_on<ArgMinImpl>(A.device(), A, axis);
}

Tensor IndexSelect(const Tensor& input, int dim, const Tensor& index) {
    if (input.device() != index.device())
        throw std::runtime_error("Tensor Device mismatch!");
    if (index.dtype() != DataType::INT32 && index.dtype() != DataType::INT64)
        throw std::runtime_error("index_select: index must be INT32 or INT64");
    if (index.shape().size() != 1)
        throw std::runtime_error("index_select: index must be 1D");
    dim = dim < 0 ? dim + static_cast<int>(input.shape().size()) : dim;
    if (dim < 0 || dim >= static_cast<int>(input.shape().size()))
        throw std::runtime_error("index_select: dim out of range");
    require_cpu(input, "index_select");
    return IndexSelectImpl<Device::CPU>::execute(input, dim, index);
}

Tensor Gather(const Tensor& input, int dim, const Tensor& index) {
    if (input.device() != index.device())
        throw std::runtime_error("Tensor Device mismatch!");
    if (index.dtype() != DataType::INT32 && index.dtype() != DataType::INT64)
        throw std::runtime_error("gather: index must be INT32 or INT64");
    dim = dim < 0 ? dim + static_cast<int>(input.shape().size()) : dim;
    if (dim < 0 || dim >= static_cast<int>(input.shape().size()))
        throw std::runtime_error("gather: dim out of range");
    require_cpu(input, "gather");
    return GatherImpl<Device::CPU>::execute(input, dim, index);
}

void Temp(Tensor& t) {
    require_contiguous(t, "temp");
    if (t.data() == nullptr || t.numel() == 0) throw std::runtime_error("tensor a is null");
    require_contiguous(t, "temp");
    require_cpu(t, "temp");
    TempImpl<Device::CPU>::execute(t);
}

}  // namespace ops
