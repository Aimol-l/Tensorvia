#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <span>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorvia.hpp"
#include "tensorvia/types.hpp"
#include "tensorvia/tensor.hpp"
#include "tensorvia/factory.hpp"
#include "tensorvia/ops.hpp"



Tensor::Tensor() {
    m_meta.numel = 0;
    m_meta.device = Device::CPU;
    m_meta.shape = {};
    m_meta.dtype = DataType::FLOAT32;
    m_meta.calculate_strides();
    m_impl = nullptr;
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype) {
    m_meta.device = Device::CPU;
    m_meta.shape = shape;
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype) {
    std::vector<int64_t> shape_(shape);
    m_meta.device = Device::CPU;
    m_meta.dtype = dtype;
    m_meta.shape = shape_;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(void* ptr, std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    std::vector<int64_t> shape_(shape);
    m_meta.device = device;
    m_meta.device_index = current_device_index(device);
    m_meta.dtype = dtype;
    m_meta.shape = shape_;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(ptr, m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(void* ptr, std::vector<int64_t> shape, DataType dtype, Device device) {
    if (ptr == nullptr) throw std::runtime_error("ptr can not nullptr");
    m_meta.device = device;
    m_meta.device_index = current_device_index(device);
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(ptr, m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, Device device) {
    m_meta.device = device;
    m_meta.device_index = current_device_index(device);
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, m_meta.device_id());
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    m_meta.device = device;
    m_meta.device_index = current_device_index(device);
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, m_meta.device_id());
}

template <typename T>
Tensor::Tensor(std::vector<T>& vec, std::initializer_list<int64_t> shape) {
    if (vec.empty()) throw std::runtime_error("Cannot create a tensor from an empty vector");
    m_meta.device = Device::CPU;
    if constexpr (std::is_same_v<T, float32>) {
        m_meta.dtype = DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, float64>) {
        m_meta.dtype = DataType::FLOAT64;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        m_meta.dtype = DataType::INT64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        m_meta.dtype = DataType::INT32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        m_meta.dtype = DataType::INT16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        m_meta.dtype = DataType::INT8;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        m_meta.dtype = DataType::BFLOAT16;
    } else if constexpr (std::is_same_v<T, float16>) {
        m_meta.dtype = DataType::FLOAT16;
    }
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(), m_meta.numel, m_meta.dtype, m_meta.device_id());
}

template <typename T>
Tensor::Tensor(std::vector<T>& vec, std::vector<int64_t> shape) {
    if (vec.empty()) throw std::runtime_error("Cannot create a tensor from an empty vector");
    m_meta.device = Device::CPU;
    if constexpr (std::is_same_v<T, float32>) {
        m_meta.dtype = DataType::FLOAT32;
    } else if constexpr (std::is_same_v<T, float64>) {
        m_meta.dtype = DataType::FLOAT64;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        m_meta.dtype = DataType::INT64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        m_meta.dtype = DataType::INT32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        m_meta.dtype = DataType::INT16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        m_meta.dtype = DataType::INT8;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        m_meta.dtype = DataType::BFLOAT16;
    } else if constexpr (std::is_same_v<T, float16>) {
        m_meta.dtype = DataType::FLOAT16;
    }
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(vec.data(), m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(std::span<const int64_t> shape, DataType dtype) {
    m_meta.device = Device::CPU;
    m_meta.shape.assign(shape.begin(), shape.end());
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(m_meta.shape);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device_id());
}

Tensor::Tensor(std::span<const int64_t> shape, DataType dtype, Device device) {
    m_meta.device = device;
    m_meta.device_index = current_device_index(device);
    m_meta.shape.assign(shape.begin(), shape.end());
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(m_meta.shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, m_meta.device_id());
}

// TensorOptions 携带 dtype 与设备*实例*（FR-7），所以这是唯一能直接命名诸如
// CUDA:1 而不依赖线程私有 current device 的构造形式。
Tensor::Tensor(std::vector<int64_t> shape, const TensorOptions& options) {
    const DeviceId id = options.resolved_id();
    m_meta.device = id.kind;
    m_meta.device_index = id.index;
    m_meta.dtype = options.dtype;
    m_meta.shape = std::move(shape);
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(m_meta.shape);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, id);
}

Tensor::Tensor(std::initializer_list<int64_t> shape, const TensorOptions& options)
    : Tensor(std::vector<int64_t>(shape), options) {}

Tensor Tensor::t() {
    if (m_meta.shape.size() != 2)
        throw std::runtime_error(std::format("t: expected a 2D tensor, got {}D", m_meta.shape.size()));
    DimArray new_shape = {m_meta.shape[1], m_meta.shape[0]};
    DimArray new_strides = {m_meta.strides[1], m_meta.strides[0]};
    return _make_view(new_shape, new_strides, 0);
}

Tensor Tensor::permute(std::initializer_list<int64_t> dims) const {
    std::vector<int64_t> dims_vec(dims);
    return this->permute(dims_vec);
}

Tensor Tensor::slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const {
    size_t ndim = m_meta.shape.size();
    if (ranges.size() > ndim) {
        throw std::invalid_argument("Too many slice ranges");
    }
    DimArray new_shape = m_meta.shape;
    DimArray new_strides = m_meta.strides;
    size_t new_offset = m_meta.offset;
    for (size_t i = 0; i < ranges.size(); ++i) {
        int64_t dim_size = m_meta.shape[i];
        auto [start_raw, end_raw] = ranges[i];
        int64_t start = start_raw < 0 ? start_raw + dim_size : start_raw;
        int64_t end   = end_raw   < 0 ? end_raw   + dim_size : end_raw;
        start = std::clamp(start, INT64_C(0), dim_size);
        end   = std::clamp(end,   start,       dim_size);
        new_shape[i] = end - start;
        new_offset += static_cast<size_t>(start) * m_meta.strides[i];
    }
    return _make_view(new_shape, new_strides, new_offset);
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    if (dims.size() != m_meta.shape.size()) {
        throw std::invalid_argument(
            "permute: number of dims (" + std::to_string(dims.size()) +
            ") doesn't match tensor dim (" + std::to_string(m_meta.shape.size()) + ")"
        );
    }
    std::vector<bool> seen(dims.size(), false);
    for (int64_t dim : dims) {
        int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
        int64_t normalized = dim >= 0 ? dim : dim + ndim;
        if (normalized < 0 || normalized >= ndim) {
            throw std::invalid_argument("Dimension out of range");
        }
        if (seen[normalized]) {
            throw std::invalid_argument("Duplicate dimension in permute");
        }
        seen[normalized] = true;
    }
    DimArray new_shape;
    DimArray new_strides;
    for (int64_t dim : dims) {
        int64_t normalized = dim >= 0 ? dim : dim + static_cast<int64_t>(m_meta.shape.size());
        new_shape.push_back(m_meta.shape[normalized]);
        new_strides.push_back(m_meta.strides[normalized]);
    }
    return _make_view(new_shape, new_strides, 0);
}

Tensor Tensor::allocate(DimArray shape, DataType dtype, DeviceId id) {
    Metadata meta;
    meta.shape = std::move(shape);
    meta.dtype = dtype;
    meta.device = id.kind;
    meta.device_index = id.index;
    meta.numel = calc_numel(meta.shape);
    meta.calculate_strides();
    auto impl = create_tensor_impl(meta.numel, dtype, id);
    return Tensor(std::move(impl), std::move(meta));
}

Tensor Tensor::clone() const {
    Tensor temp = allocate(m_meta.shape, m_meta.dtype, m_meta.device_id());
    // copy_to 从底层 buffer 基址整块拷贝，只对无 offset 且正好覆盖整个
    // buffer 的张量成立；带 offset 的连续切片和任何非连续视图都走
    // clone_as_contiguous 物化。
    if (this->is_contiguous() && m_meta.offset == 0 &&
        m_impl->numel() == static_cast<size_t>(m_meta.numel)) {
        this->m_impl->copy_to(temp.m_impl);
        return temp;
    }
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    temp.m_impl = std::move(new_impl);
    return temp;
}

// data() 返回视图首元素（含 offset），而非底层 buffer 起点：
// slice/t()/permute 产生的视图共享 impl，kernel 从 data() 起线性遍历
// numel 个元素才能命中视图自己的数据。
void* Tensor::data() {
    return static_cast<char*>(m_impl->data())
         + static_cast<size_t>(m_meta.offset) * calc_dtype_size(m_meta.dtype);
}
const void* Tensor::data() const {
    return static_cast<const char*>(m_impl->data())
         + static_cast<size_t>(m_meta.offset) * calc_dtype_size(m_meta.dtype);
}

Tensor Tensor::contiguous() const {
    if (this->is_contiguous()) return *this;
    Tensor contig_tensor = allocate(m_meta.shape, m_meta.dtype, m_meta.device_id());
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    contig_tensor.m_impl = std::move(new_impl);
    return contig_tensor;
}

bool Tensor::is_contiguous() const { return m_meta.is_contig; }

Tensor& Tensor::to_contiguous() {
    if (this->is_contiguous()) return *this;
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    this->m_impl = std::move(new_impl);
    // 物化后的 impl 拥有从首元素开始的新 buffer，所以旧的视图 offset 不能再叠加到
    // 它上面。
    m_meta.offset = 0;
    m_meta.calculate_strides();
    return *this;
}

void Tensor::materialize_for_transfer() {
    // 连续且无 offset 还不够：前导维的视图（narrow/split/select）从 offset 0 起连续，
    // 但覆盖的元素少于它指向的 buffer。传输会整块读取 impl buffer，所以这种视图需要
    // 一个按自身大小分配的独立 buffer。
    if (m_meta.is_contig && m_meta.offset == 0 &&
        m_impl->numel() == static_cast<size_t>(m_meta.numel)) return;
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    m_impl = std::move(new_impl);
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

Tensor Tensor::view(std::initializer_list<int64_t> new_shape_list) const {
    std::vector<int64_t> new_shape(new_shape_list);
    return this->view(new_shape);
}

Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    if (!this->is_contiguous()) {
        throw std::invalid_argument("view: tensor is not contiguous. Call .contiguous() first.");
    }
    int64_t infer_dim = -1;
    int64_t new_numel = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("Only one dimension can be -1");
            }
            infer_dim = static_cast<int64_t>(i);
            continue;
        }
        if (new_shape[i] < 0) {
            throw std::invalid_argument("Negative dimension size not allowed (except -1)");
        }
        new_numel *= new_shape[i];
    }
    if (infer_dim != -1) {
        if (new_numel == 0 || this->numel() % new_numel != 0) {
            throw std::invalid_argument("view: invalid inferred size for -1 (numel mismatch)");
        }
        new_shape[infer_dim] = static_cast<int64_t>(this->numel() / new_numel);
        new_numel = this->numel();
    }
    if (new_numel != static_cast<int64_t>(this->numel())) {
        throw std::invalid_argument("view: new shape numel does not match original numel");
    }
    DimArray new_strides(new_shape.size());
    if (!new_shape.empty()) {
        new_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(new_shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
    }
    // 带 offset 的视图（切片）保留其 offset：新步长描述的是同一个首元素，而它并不是
    // buffer 基址。
    return this->_make_view(new_shape, new_strides, m_meta.offset);
}

Tensor Tensor::_make_view(DimArray shape, DimArray strides, int64_t offset) const {
    Metadata meta;
    meta.numel = calc_numel(shape);
    meta.dtype = this->m_meta.dtype;
    meta.device = this->m_meta.device;
    meta.device_index = this->m_meta.device_index;
    meta.offset = offset;
    meta.shape = std::move(shape);
    meta.strides = std::move(strides);
    meta.update_contiguous();
    return Tensor(this->m_impl, std::move(meta));
}

// 引用语义（对齐 NumPy/PyTorch）：拷贝 = O(1) 共享底层 buffer 的句柄复制。
// 数据由 shared_ptr<TensorImpl> 引用计数管理；shape/strides 等 meta 各自独立
// （b = a 后 reshape/slice b 不影响 a 的形状，比 NumPy 的共享元数据更安全）。
// 显式深拷贝用 clone()。
Tensor::Tensor(const Tensor& other)
    : m_meta(other.m_meta), m_impl(other.m_impl) {}

Tensor::Tensor(Tensor&& other) noexcept {
    this->m_meta = std::move(other.m_meta);
    this->m_impl = std::move(other.m_impl);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        m_meta = other.m_meta;
        m_impl = other.m_impl;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        m_meta = std::move(other.m_meta);
        m_impl = std::move(other.m_impl);
    }
    return *this;
}

Device Tensor::device() const { return m_meta.device; }

int Tensor::device_index() const { return m_meta.device_index; }

DeviceId Tensor::device_id() const { return {m_meta.device, m_meta.device_index}; }

std::span<const int64_t> Tensor::shape() const { return {m_meta.shape.data(), m_meta.shape.size()}; }

// 源连续时 reshape 是视图；否则先物化成连续拷贝（PyTorch 的规则），这也是跨步视图
// 能被重新解释的唯一办法。源永远不会被修改。
Tensor Tensor::reshape(std::vector<int64_t> newshape) const {
    if (is_contiguous()) return view(std::move(newshape));
    Tensor copy = contiguous();
    return copy.view(std::move(newshape));
}

Tensor Tensor::reshape(std::initializer_list<int64_t> newshape) const {
    std::vector<int64_t> shape(newshape);
    return reshape(std::move(shape));
}

size_t Tensor::dims() const { return m_meta.shape.size(); }

int64_t Tensor::shape(int i) {
    if (i < 0) return m_meta.shape[m_meta.shape.size() + i];
    return m_meta.shape[i];
}

int64_t Tensor::shape(int i) const {
    if (i < 0) return m_meta.shape[m_meta.shape.size() + i];
    return m_meta.shape[i];
}

DataType Tensor::dtype() const { return m_meta.dtype; }

Tensor Tensor::to_type(DataType dst) const {
    return ops::Typecast(static_cast<const Tensor&>(*this), dst);
}

void Tensor::to_type_(DataType dst) {
    ops::Typecast(*this, dst);
}

// 显式原地系列（FR-6）：尾随下划线标明这是修改操作，自由函数 ops:: 的非 const
// 重载为兼容而保留。它们都写穿共享存储，因此别名也能看到变化。
Tensor& Tensor::fill_(float value) { ops::Fill(*this, value); return *this; }
Tensor& Tensor::zero_() { ops::Zeros(*this); return *this; }
Tensor& Tensor::add_(float value) { ops::Add(*this, value); return *this; }
Tensor& Tensor::sub_(float value) { ops::Sub(*this, value); return *this; }
Tensor& Tensor::mul_(float value) { ops::Mul(*this, value); return *this; }
Tensor& Tensor::div_(float value) { ops::Div(*this, value); return *this; }
Tensor& Tensor::abs_() { ops::Abs(*this); return *this; }
Tensor& Tensor::relu_() { ops::Relu(*this); return *this; }
Tensor& Tensor::clamp_(float min, float max) { ops::Clamp(*this, min, max); return *this; }

Tensor& Tensor::copy_(const Tensor& src) {
    if (src.numel() != numel())
        throw std::runtime_error(std::format(
            "copy_: source has {} elements, destination has {}", src.numel(), numel()));
    // 不覆盖整个 buffer 的目标无法写穿：后端写入从 buffer 基址开始，部分视图要么
    // 越界（设备），要么静默写到错误元素上。所有后端规则一致。
    if (m_meta.offset != 0 || m_impl->numel() != static_cast<size_t>(m_meta.numel))
        throw std::runtime_error(
            "copy_: destination is a view that does not cover its whole buffer; clone it first");
    // 在宿主以本张量的 dtype 暂存；相对于这次调用隐含的工作量，元素数很少，而且
    // 这样每种（设备, dtype）组合都能共用一条代码路径。
    Tensor staged = (src.dtype() == m_meta.dtype) ? src.clone() : ops::Typecast(src, m_meta.dtype);
    staged.to_host();
    if (m_meta.device == Device::CPU) {
        if (!is_contiguous())
            throw std::runtime_error("copy_: destination is a non-contiguous view; call to_contiguous() first");
        std::memcpy(this->data(), staged.data(), numel() * calc_dtype_size(m_meta.dtype));
        return *this;
    }
    m_impl->copy_from_host(staged.data(), m_meta.dtype);
    return *this;
}

void Tensor::to_host() {
    if (m_meta.device == Device::CPU) return;
    // copy_to_host 基于 impl 的原始基址指针工作，所以先把任何视图（offset 和/或
    // 非连续步长）折叠成独立 buffer。
    materialize_for_transfer();
    auto cpu_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, DeviceId{Device::CPU, 0});
    m_impl->copy_to_host(cpu_impl->data(), m_meta.dtype);
    m_impl = std::move(cpu_impl);
    m_meta.device = Device::CPU;
    m_meta.device_index = 0;
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

void Tensor::to_device(DeviceId target) {
    if (target.index < 0)
        throw std::invalid_argument("to_device: device index must be non-negative");
    if (target.kind == Device::CPU && target.index != 0)
        throw std::invalid_argument("to_device: the CPU backend has only index 0");
    if (m_meta.device_id() == target) return;
    if (target.kind == Device::CPU) {
        to_host();
        return;
    }
    // 非 CPU -> 非 CPU 的搬移很少见；经宿主机往返让传输逻辑集中在一处（而且正确性上
    // 本就要求如此）。
    if (m_meta.device != Device::CPU) to_host();
    materialize_for_transfer();
    auto device_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, target);
    device_impl->copy_from_host(m_impl->data(), m_meta.dtype);
    m_impl = std::move(device_impl);
    m_meta.device = target.kind;
    m_meta.device_index = target.index;
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

void Tensor::to_device(Device target, int index) {
    to_device(DeviceId{target, index < 0 ? current_device_index(target) : index});
}

void Tensor::reset(std::vector<int64_t> shape, DataType dtype) {
    size_t new_numel   = calc_numel(shape);
    size_t old_numel   = m_meta.numel;
    DataType old_dtype = m_meta.dtype;

    // 复用 m_meta.shape / m_meta.strides 的 vector 容量（assign 不重分配 capacity 足够时）。
    m_meta.shape.assign(shape.begin(), shape.end());
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = new_numel;

    // buffer 复用条件：m_impl 唯一持有 + 字节大小完全匹配（同 numel + 同 dtype）。
    // 满足时连 pool allocate 都省了；否则让 factory 走 pool 重新分配。
    bool reuse_impl = m_impl
                   && m_impl.use_count() == 1
                   && old_dtype == dtype
                   && old_numel == new_numel;

    if (!reuse_impl) {
        m_impl = create_tensor_impl(new_numel, dtype, m_meta.device_id());
    }
}

void Tensor::reset(std::initializer_list<int64_t> shape, DataType dtype) {
    std::vector<int64_t> v(shape);
    this->reset(std::move(v), dtype);
}

int64_t Tensor::strides(int i) const { return m_meta.strides[i]; }
std::span<const int64_t> Tensor::strides() const { return {m_meta.strides.data(), m_meta.strides.size()}; }

Tensor Tensor::empty_like(Tensor& tensor) const {
    // 形状与设备跟随 `tensor`，只有 dtype 来自 self。
    DimArray shape_of_source;
    shape_of_source.assign(tensor.shape().begin(), tensor.shape().end());
    return allocate(shape_of_source, m_meta.dtype, tensor.device_id());
}

void Tensor::set_impl(std::shared_ptr<TensorImpl> impl, DataType new_dtype) {
    m_impl = std::move(impl);
    m_meta.dtype = new_dtype;
    // 新 impl 是全新的连续 buffer；旧存储的视图元数据不再适用。
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

std::shared_ptr<TensorImpl> Tensor::get_impl() const { return m_impl; }

Tensor Tensor::operator+(const Tensor& other) const { return ops::Add(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return ops::Sub(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return ops::Mul(*this, other); }

Tensor operator+(float a, const Tensor& other) { return ops::Add(other, a); }
Tensor Tensor::operator+(float a) const { return ops::Add(*this, a); }
Tensor operator-(float a, const Tensor& other) { return ops::Sub(other, a); }
Tensor Tensor::operator-(float a) const { return ops::Sub(*this, a); }
Tensor operator*(float a, const Tensor& other) { return ops::Mul(other, a); }
Tensor Tensor::operator*(float a) const { return ops::Mul(*this, a); }
Tensor Tensor::operator/(const Tensor& other) const { return ops::Div(*this, other); }
Tensor Tensor::operator/(float a) const { return ops::Div(*this, a); }

Tensor Tensor::Zeros(std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    std::vector<int64_t> shape_(shape);
    return ops::Zeros(shape_, dtype, device);
}
Tensor Tensor::Zeros(std::vector<int64_t> shape, DataType dtype, Device device) {
    return ops::Zeros(shape, dtype, device);
}
Tensor Tensor::Ones(std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    std::vector<int64_t> shape_(shape);
    return ops::Ones(shape_, dtype, device);
}
Tensor Tensor::Ones(std::vector<int64_t> shape, DataType dtype, Device device) {
    return ops::Ones(shape, dtype, device);
}
Tensor Tensor::Fill(std::initializer_list<int64_t> shape, float value, DataType dtype, Device device) {
    std::vector<int64_t> shape_(shape);
    return ops::Fill(shape_, dtype, value, device);
}
Tensor Tensor::Fill(std::vector<int64_t> shape, float value, DataType dtype, Device device) {
    return ops::Fill(shape, dtype, value, device);
}
Tensor Tensor::Random(std::initializer_list<int64_t> shape, float min, float max, DataType dtype, Device device) {
    std::vector<int64_t> shape_(shape);
    return ops::Random(shape_, dtype, min, max, device);
}
Tensor Tensor::Random(std::vector<int64_t> shape, float min, float max, DataType dtype, Device device) {
    return ops::Random(shape, dtype, min, max, device);
}

Tensor Tensor::empty(std::vector<int64_t> shape, const TensorOptions& options) {
    return Tensor(std::move(shape), options);
}
Tensor Tensor::empty(std::initializer_list<int64_t> shape, const TensorOptions& options) {
    return Tensor(shape, options);
}
Tensor Tensor::Zeros(std::vector<int64_t> shape, const TensorOptions& options) {
    return ops::Zeros(shape, options);
}
Tensor Tensor::Zeros(std::initializer_list<int64_t> shape, const TensorOptions& options) {
    return ops::Zeros(std::vector<int64_t>(shape), options);
}
Tensor Tensor::Ones(std::vector<int64_t> shape, const TensorOptions& options) {
    return ops::Ones(shape, options);
}
Tensor Tensor::Ones(std::initializer_list<int64_t> shape, const TensorOptions& options) {
    return ops::Ones(std::vector<int64_t>(shape), options);
}
Tensor Tensor::Fill(std::vector<int64_t> shape, float value, const TensorOptions& options) {
    return ops::Fill(shape, value, options);
}
Tensor Tensor::Fill(std::initializer_list<int64_t> shape, float value, const TensorOptions& options) {
    return ops::Fill(std::vector<int64_t>(shape), value, options);
}
Tensor Tensor::Random(std::vector<int64_t> shape, float min, float max, const TensorOptions& options) {
    return ops::Random(shape, min, max, options);
}
Tensor Tensor::Random(std::initializer_list<int64_t> shape, float min, float max,
                      const TensorOptions& options) {
    return ops::Random(std::vector<int64_t>(shape), min, max, options);
}

template <typename T>
T Tensor::at(std::initializer_list<int64_t> idxs) {
    if (this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if (idxs.size() != m_meta.shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for (size_t i = 0; i < idxs_.size(); i++) {
        if (idxs_[i] >= this->shape(i) || idxs_[i] < 0) throw std::runtime_error("index out of range");
    }
    // data() 已含视图 offset，这里只按 strides 定位，否则双重偏移。
    size_t index = 0;
    for (size_t i = 0; i < idxs_.size(); i++) {
        index += idxs_[i] * m_meta.strides[i];
    }
    switch (m_meta.dtype) {
        case DataType::BOOL:     return static_cast<T>(static_cast<uint8_t*>(this->data())[index] != 0);
        case DataType::INT8:     return static_cast<T>(static_cast<int8_t*>(this->data())[index]);
        case DataType::INT16:    return static_cast<T>(static_cast<int16_t*>(this->data())[index]);
        case DataType::INT32:    return static_cast<T>(static_cast<int32_t*>(this->data())[index]);
        case DataType::INT64:    return static_cast<T>(static_cast<int64_t*>(this->data())[index]);
        case DataType::FLOAT16:  return static_cast<T>(static_cast<float16*>(this->data())[index]);
        case DataType::BFLOAT16: return static_cast<T>(static_cast<bfloat16*>(this->data())[index]);
        case DataType::FLOAT32:  return static_cast<T>(static_cast<float32*>(this->data())[index]);
        case DataType::FLOAT64:  return static_cast<T>(static_cast<float64*>(this->data())[index]);
        default: throw std::runtime_error("Unsupported dtype in Tensor::at");
    }
}

template <typename T>
T Tensor::operator[](std::initializer_list<int64_t> idxs) {
    if (this->device() != Device::CPU) throw std::runtime_error("Tensor device must be CPU");
    if (idxs.size() != m_meta.shape.size()) throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for (size_t i = 0; i < idxs_.size(); i++) {
        if (idxs_[i] >= this->shape(i) || idxs_[i] < 0) throw std::runtime_error("index out of range");
    }
    // data() 已含视图 offset，这里只按 strides 定位，否则双重偏移。
    size_t index = 0;
    for (size_t i = 0; i < idxs_.size(); i++) {
        index += idxs_[i] * m_meta.strides[i];
    }
    switch (m_meta.dtype) {
        case DataType::BOOL:     return static_cast<T>(static_cast<uint8_t*>(this->data())[index] != 0);
        case DataType::INT8:     return static_cast<T>(static_cast<int8_t*>(this->data())[index]);
        case DataType::INT16:    return static_cast<T>(static_cast<int16_t*>(this->data())[index]);
        case DataType::INT32:    return static_cast<T>(static_cast<int32_t*>(this->data())[index]);
        case DataType::INT64:    return static_cast<T>(static_cast<int64_t*>(this->data())[index]);
        case DataType::FLOAT16:  return static_cast<T>(static_cast<float16*>(this->data())[index]);
        case DataType::BFLOAT16: return static_cast<T>(static_cast<bfloat16*>(this->data())[index]);
        case DataType::FLOAT32:  return static_cast<T>(static_cast<float32*>(this->data())[index]);
        case DataType::FLOAT64:  return static_cast<T>(static_cast<float64*>(this->data())[index]);
        default: throw std::runtime_error("Unsupported dtype in Tensor::at");
    }
}

// 是视图而非修改（FR-4）：squeeze/unsqueeze 用少/多一维来描述同一存储，与
// slice/permute 完全一样。两个操作数都不会被修改。
Tensor Tensor::squeeze() const {
    DimArray shape, strides;
    for (size_t i = 0; i < m_meta.shape.size(); ++i) {
        if (m_meta.shape[i] != 1) {
            shape.push_back(m_meta.shape[i]);
            strides.push_back(m_meta.strides[i]);
        }
    }
    return _make_view(shape, strides, m_meta.offset);
}

Tensor Tensor::squeeze(int64_t dim) const {
    const int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error(std::format(
            "squeeze: dim {} is out of range for a {}D tensor", dim, ndim));
    if (m_meta.shape[dim] != 1) return *this;  // 无需删除：返回同一视图
    DimArray shape, strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        shape.push_back(m_meta.shape[i]);
        strides.push_back(m_meta.strides[i]);
    }
    return _make_view(shape, strides, m_meta.offset);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    const int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim)
        throw std::runtime_error(std::format(
            "unsqueeze: dim {} is out of range for a {}D tensor", dim, ndim));
    DimArray shape, strides;
    for (int64_t i = 0; i <= ndim; ++i) {
        if (i == dim) {
            shape.push_back(1);
            // 新维重复单个位置：步长取其后一维的跨度（末尾为 1）。
            strides.push_back(dim < ndim ? m_meta.strides[dim] * m_meta.shape[dim] : 1);
            continue;
        }
        // 插入点之后的源维整体左移一位。
        const int64_t src = i < dim ? i : i - 1;
        shape.push_back(m_meta.shape[src]);
        strides.push_back(m_meta.strides[src]);
    }
    return _make_view(shape, strides, m_meta.offset);
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    const int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error(std::format(
            "narrow: dim {} is out of range for a {}D tensor", dim, ndim));
    const int64_t size = m_meta.shape[dim];
    if (start < 0) start += size;
    if (start < 0 || start > size)
        throw std::runtime_error(std::format(
            "narrow: start {} is out of range for dimension {} of size {}", start, dim, size));
    if (length < 0 || start + length > size)
        throw std::runtime_error(std::format(
            "narrow: length {} from start {} exceeds dimension {} of size {}", length, start, dim, size));
    DimArray shape = m_meta.shape;
    shape[dim] = length;
    const int64_t offset = m_meta.offset + start * m_meta.strides[dim];
    return _make_view(shape, m_meta.strides, offset);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    const int64_t ndim = static_cast<int64_t>(m_meta.shape.size());
    if (ndim == 0)
        throw std::runtime_error("select: a 0-dim tensor has no dimension to select from");
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::runtime_error(std::format(
            "select: dim {} is out of range for a {}D tensor", dim, ndim));
    const int64_t size = m_meta.shape[dim];
    if (index < 0) index += size;
    if (index < 0 || index >= size)
        throw std::runtime_error(std::format(
            "select: index {} is out of range for dimension {} of size {}", index, dim, size));
    DimArray shape, strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        shape.push_back(m_meta.shape[i]);
        strides.push_back(m_meta.strides[i]);
    }
    const int64_t offset = m_meta.offset + index * m_meta.strides[dim];
    return _make_view(shape, strides, offset);
}

Tensor Tensor::expand(std::initializer_list<int64_t> shape) const {
    return expand(std::vector<int64_t>(shape));
}

Tensor Tensor::expand_as(const Tensor& other) const {
    return expand(std::vector<int64_t>(other.shape().begin(), other.shape().end()));
}

Tensor Tensor::expand(std::vector<int64_t> shape) const {
    const size_t rank = m_meta.shape.size();
    if (shape.size() < rank)
        throw std::runtime_error(std::format(
            "expand: target rank {} is smaller than the tensor rank {}", shape.size(), rank));
    DimArray new_shape, new_strides;
    const size_t lead = shape.size() - rank;
    for (size_t i = 0; i < shape.size(); ++i) {
        const int64_t target = shape[i];
        if (target < 0)
            throw std::runtime_error(std::format("expand: negative size {} at dimension {}", target, i));
        if (i < lead) {
            // 新前导维无物可指：步长为 0。
            new_shape.push_back(target);
            new_strides.push_back(0);
            continue;
        }
        const size_t src = i - lead;
        if (m_meta.shape[src] == target) {
            new_shape.push_back(target);
            new_strides.push_back(m_meta.strides[src]);
        } else if (m_meta.shape[src] == 1) {
            // 长度 1 的维可重复任意次：步长为 0。
            new_shape.push_back(target);
            new_strides.push_back(0);
        } else {
            throw std::runtime_error(std::format(
                "expand: dimension {} of size {} cannot be expanded to {}", src,
                m_meta.shape[src], target));
        }
    }
    return _make_view(new_shape, new_strides, m_meta.offset);
}

Tensor Tensor::operator==(const Tensor& other) const { return ops::Equal(*this, other); }
Tensor Tensor::operator==(const float val) const {
    return ops::Equal(*this, val);
}
Tensor Tensor::operator!=(const Tensor& other) const { return ops::NotEqual(*this, other); }
Tensor Tensor::operator!=(const float val) const {
    return ops::NotEqual(*this, val);
}
Tensor Tensor::operator>(const Tensor& other) const { return ops::Greater(*this, other); }
Tensor Tensor::operator>(const float val) const {
    return ops::Greater(*this, val);
}
Tensor Tensor::operator>=(const Tensor& other) const { return ops::GreaterEqual(*this, other); }
Tensor Tensor::operator>=(const float val) const {
    return ops::GreaterEqual(*this, val);
}
Tensor Tensor::operator<(const Tensor& other) const { return ops::Less(*this, other); }
Tensor Tensor::operator<(const float val) const {
    return ops::Less(*this, val);
}
Tensor Tensor::operator<=(const Tensor& other) const { return ops::LessEqual(*this, other); }
Tensor Tensor::operator<=(const float val) const {
    return ops::LessEqual(*this, val);
}

template <typename T>
void Tensor::set(std::initializer_list<int64_t> idxs, T value) {
    if (this->device() != Device::CPU)
        throw std::runtime_error("set: the tensor must be on the CPU");
    if (idxs.size() != m_meta.shape.size())
        throw std::runtime_error("Tensor index size must be equal to tensor shape size");
    std::vector<int64_t> idxs_(idxs);
    for (size_t i = 0; i < idxs_.size(); ++i)
        if (idxs_[i] >= this->shape(i) || idxs_[i] < 0)
            throw std::runtime_error("index out of range");
    // 与 at() 相同的布局规则：data() 已含视图 offset，所以仅靠步长即可定位元素
    // ——通过视图写入会落到它别名的那块存储上。
    size_t index = 0;
    for (size_t i = 0; i < idxs_.size(); ++i) index += idxs_[i] * m_meta.strides[i];
    switch (m_meta.dtype) {
    case DataType::BOOL:     static_cast<uint8_t*>(this->data())[index] = static_cast<bool>(value) ? 1 : 0; break;
    case DataType::INT8:     static_cast<int8_t*>(this->data())[index] = static_cast<int8_t>(value); break;
    case DataType::INT16:    static_cast<int16_t*>(this->data())[index] = static_cast<int16_t>(value); break;
    case DataType::INT32:    static_cast<int32_t*>(this->data())[index] = static_cast<int32_t>(value); break;
    case DataType::INT64:    static_cast<int64_t*>(this->data())[index] = static_cast<int64_t>(value); break;
    case DataType::FLOAT16:  static_cast<float16*>(this->data())[index] = static_cast<float16>(value); break;
    case DataType::BFLOAT16: static_cast<bfloat16*>(this->data())[index] = static_cast<bfloat16>(value); break;
    case DataType::FLOAT32:  static_cast<float32*>(this->data())[index] = static_cast<float32>(value); break;
    case DataType::FLOAT64:  static_cast<float64*>(this->data())[index] = static_cast<float64>(value); break;
    default: throw std::runtime_error("Unsupported dtype in Tensor::set");
    }
}

template <typename T>
T Tensor::item() const {
    if (this->numel() != 1)
        throw std::runtime_error(std::format(
            "item: tensor must hold exactly one element, got numel={}", this->numel()));
    if (m_impl == nullptr) throw std::runtime_error("item: tensor is null");
    // 任意设备上的单个元素：只需经宿主内存中转这几个字节。在设备后端这是同步拷贝
    // ——也是归约结果回到宿主的唯一位置。
    // copy_to_host 从 impl 的 buffer 基址读取，所以单元素视图（offset 和/或步长）
    // 必须先物化；在局部句柄上做，以保持 item() 为 const。
    Tensor src = *this;
    src.materialize_for_transfer();
    alignas(16) unsigned char buf[16] = {};
    src.m_impl->copy_to_host(buf, src.m_meta.dtype);
    switch (m_meta.dtype) {
        case DataType::BOOL: {
            uint8_t v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v != 0);
        }
        case DataType::INT8: {
            int8_t v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::INT16: {
            int16_t v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::INT32: {
            int32_t v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::INT64: {
            int64_t v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::FLOAT16: {
            float16 v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::BFLOAT16: {
            bfloat16 v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::FLOAT32: {
            float32 v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        case DataType::FLOAT64: {
            float64 v; std::memcpy(&v, buf, sizeof v);
            return static_cast<T>(v);
        }
        default: throw std::runtime_error("Unsupported dtype in Tensor::item");
    }
}

template void Tensor::set<bool>(std::initializer_list<int64_t> idxs, bool value);
template void Tensor::set<int8_t>(std::initializer_list<int64_t> idxs, int8_t value);
template void Tensor::set<int16_t>(std::initializer_list<int64_t> idxs, int16_t value);
template void Tensor::set<int32_t>(std::initializer_list<int64_t> idxs, int32_t value);
template void Tensor::set<int64_t>(std::initializer_list<int64_t> idxs, int64_t value);
template void Tensor::set<float16>(std::initializer_list<int64_t> idxs, float16 value);
template void Tensor::set<bfloat16>(std::initializer_list<int64_t> idxs, bfloat16 value);
template void Tensor::set<float32>(std::initializer_list<int64_t> idxs, float32 value);
template void Tensor::set<float64>(std::initializer_list<int64_t> idxs, float64 value);

template bool Tensor::at<bool>(std::initializer_list<int64_t> idxs);
template int8_t Tensor::at<int8_t>(std::initializer_list<int64_t> idxs);
template int16_t Tensor::at<int16_t>(std::initializer_list<int64_t> idxs);
template int32_t Tensor::at<int32_t>(std::initializer_list<int64_t> idxs);
template int64_t Tensor::at<int64_t>(std::initializer_list<int64_t> idxs);
template float16 Tensor::at<float16>(std::initializer_list<int64_t> idxs);
template bfloat16 Tensor::at<bfloat16>(std::initializer_list<int64_t> idxs);
template float32 Tensor::at<float32>(std::initializer_list<int64_t> idxs);
template float64 Tensor::at<float64>(std::initializer_list<int64_t> idxs);

template bool Tensor::item<bool>() const;
template int8_t Tensor::item<int8_t>() const;
template int16_t Tensor::item<int16_t>() const;
template int32_t Tensor::item<int32_t>() const;
template int64_t Tensor::item<int64_t>() const;
template float16 Tensor::item<float16>() const;
template bfloat16 Tensor::item<bfloat16>() const;
template float32 Tensor::item<float32>() const;
template float64 Tensor::item<float64>() const;

template bool Tensor::operator[]<bool>(std::initializer_list<int64_t> idxs);
template int8_t Tensor::operator[]<int8_t>(std::initializer_list<int64_t> idxs);
template int16_t Tensor::operator[]<int16_t>(std::initializer_list<int64_t> idxs);
template int32_t Tensor::operator[]<int32_t>(std::initializer_list<int64_t> idxs);
template int64_t Tensor::operator[]<int64_t>(std::initializer_list<int64_t> idxs);
template float16 Tensor::operator[]<float16>(std::initializer_list<int64_t> idxs);
template bfloat16 Tensor::operator[]<bfloat16>(std::initializer_list<int64_t> idxs);
template float32 Tensor::operator[]<float32>(std::initializer_list<int64_t> idxs);
template float64 Tensor::operator[]<float64>(std::initializer_list<int64_t> idxs);

template Tensor::Tensor(std::vector<int8_t>&,  std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<int16_t>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<int32_t>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<int64_t>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<float16>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<bfloat16>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<float32>&, std::initializer_list<int64_t>);
template Tensor::Tensor(std::vector<float64>&, std::initializer_list<int64_t>);

template Tensor::Tensor(std::vector<int8_t>&,  std::vector<int64_t>);
template Tensor::Tensor(std::vector<int16_t>&, std::vector<int64_t>);
template Tensor::Tensor(std::vector<int32_t>&, std::vector<int64_t>);
template Tensor::Tensor(std::vector<int64_t>&, std::vector<int64_t>);
template Tensor::Tensor(std::vector<float16>&, std::vector<int64_t>);
template Tensor::Tensor(std::vector<bfloat16>&, std::vector<int64_t>);
template Tensor::Tensor(std::vector<float32>&,  std::vector<int64_t>);
template Tensor::Tensor(std::vector<float64>&,  std::vector<int64_t>);
