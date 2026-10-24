#include <algorithm>
#include <cstdint>
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
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    m_meta.device = Device::CPU;
    m_meta.shape = shape;
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    m_meta.device = Device::CPU;
    m_meta.dtype = dtype;
    m_meta.shape = shape_;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(void* ptr, std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    m_meta.device = device;
    m_meta.dtype = dtype;
    m_meta.shape = shape_;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape_);
    m_impl = create_tensor_impl(ptr, m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(void* ptr, std::vector<int64_t> shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    if (ptr == nullptr) throw std::runtime_error("ptr can not nullptr");
    m_meta.device = device;
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(ptr, m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    m_meta.device = device;
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, device);
}

Tensor::Tensor(std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    m_meta.device = device;
    m_meta.dtype = dtype;
    m_meta.shape = shape;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, device);
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
    m_impl = create_tensor_impl(vec.data(), m_meta.numel, m_meta.dtype, m_meta.device);
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
    m_impl = create_tensor_impl(vec.data(), m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(std::span<const int64_t> shape, DataType dtype) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    m_meta.device = Device::CPU;
    m_meta.shape.assign(shape.begin(), shape.end());
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(m_meta.shape);
    m_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, m_meta.device);
}

Tensor::Tensor(std::span<const int64_t> shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    m_meta.device = device;
    m_meta.shape.assign(shape.begin(), shape.end());
    m_meta.dtype = dtype;
    m_meta.calculate_strides();
    m_meta.numel = calc_numel(m_meta.shape);
    m_impl = create_tensor_impl(m_meta.numel, dtype, device);
}

Tensor Tensor::t() {
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

Tensor Tensor::clone() const {
    Tensor temp(this->m_meta.shape, this->m_meta.dtype, this->m_meta.device);
    // copy_to 从底层 buffer 基址整块拷贝，只对无 offset 的张量成立；
    // 带 offset 的连续切片和任何非连续视图都走 clone_as_contiguous 物化。
    if (this->is_contiguous() && m_meta.offset == 0) {
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
    Tensor contig_tensor(m_meta.shape, m_meta.dtype, m_meta.device);
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    contig_tensor.m_impl = std::move(new_impl);
    return contig_tensor;
}

bool Tensor::is_contiguous() const { return m_meta.is_contig; }

Tensor& Tensor::to_contiguous() {
    if (this->is_contiguous()) return *this;
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    this->m_impl = std::move(new_impl);
    // The materialized impl owns a fresh buffer starting at its first element,
    // so the old view offset must not be applied to it again.
    m_meta.offset = 0;
    m_meta.calculate_strides();
    return *this;
}

void Tensor::materialize_for_transfer() {
    if (m_meta.is_contig && m_meta.offset == 0) return;
    auto new_impl = m_impl->clone_as_contiguous(m_meta);
    m_impl = std::move(new_impl);
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

Tensor Tensor::view(std::initializer_list<int64_t> new_shape_list) {
    std::vector<int64_t> new_shape(new_shape_list);
    return this->view(new_shape);
}

Tensor Tensor::view(std::vector<int64_t> new_shape) {
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
    return this->_make_view(new_shape, new_strides, 0);
}

Tensor Tensor::_make_view(DimArray shape, DimArray strides, int64_t offset) const {
    Metadata meta;
    meta.numel = calc_numel(shape);
    meta.dtype = this->m_meta.dtype;
    meta.device = this->m_meta.device;
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

std::span<const int64_t> Tensor::shape() const { return {m_meta.shape.data(), m_meta.shape.size()}; }

void Tensor::reshape(std::vector<int64_t>& newshape) {
    int old_total = calc_numel(this->m_meta.shape);
    int new_total = calc_numel(newshape);
    if (old_total != new_total) {
        std::string info = std::format("new elements count must be equal to old elements count. {} != {}", old_total, new_total);
        throw std::runtime_error(info);
    }
    m_meta.shape.clear();
    m_meta.shape.assign(newshape.begin(), newshape.end());
    m_meta.calculate_strides();
}

void Tensor::reshape(std::initializer_list<int64_t> newshape) {
    int old_total = calc_numel(this->m_meta.shape);
    int new_total = calc_numel(newshape);
    if (old_total != new_total) {
        throw std::runtime_error("new elements count must be equal to old elements count");
    }
    m_meta.shape.clear();
    m_meta.shape.assign(newshape.begin(), newshape.end());
    m_meta.calculate_strides();
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

void Tensor::to_host() {
    if (m_meta.device == Device::CPU) return;
    // copy_to_host works on the impl's raw base pointer, so collapse any view
    // (offset and/or non-contiguous strides) into a standalone buffer first.
    materialize_for_transfer();
    auto cpu_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, Device::CPU);
    m_impl->copy_to_host(cpu_impl->data(), m_meta.dtype);
    m_impl = std::move(cpu_impl);
    m_meta.device = Device::CPU;
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

void Tensor::to_device(Device target) {
    if (m_meta.device == target) return;
    if (target == Device::CPU) {
        to_host();
        return;
    }
    // A non-CPU -> non-CPU move is rare; round-trip through host keeps the
    // transfer logic in one place (and is required for correctness anyway).
    if (m_meta.device != Device::CPU) to_host();
    materialize_for_transfer();
    auto device_impl = create_tensor_impl(m_meta.numel, m_meta.dtype, target);
    device_impl->copy_from_host(m_impl->data(), m_meta.dtype);
    m_impl = std::move(device_impl);
    m_meta.device = target;
    m_meta.offset = 0;
    m_meta.calculate_strides();
}

void Tensor::to_device(uint32_t) {
#if defined(BACKEND_VULKAN)
    to_device(Device::VULKAN);
#elif defined(BACKEND_CUDA)
    to_device(Device::CUDA);
#elif defined(BACKEND_SYCL)
    to_device(Device::SYCL);
#endif
    // CPU-only build: no accelerator to move to.
}

void Tensor::reset(std::vector<int64_t> shape, DataType dtype) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
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
        m_impl = create_tensor_impl(new_numel, dtype, m_meta.device);
    }
}

void Tensor::reset(std::initializer_list<int64_t> shape, DataType dtype) {
    std::vector<int64_t> v(shape);
    this->reset(std::move(v), dtype);
}

int64_t Tensor::strides(int i) const { return m_meta.strides[i]; }
std::span<const int64_t> Tensor::strides() const { return {m_meta.strides.data(), m_meta.strides.size()}; }

Tensor Tensor::empty_like(Tensor& tensor) const {
    return Tensor(tensor.shape(), m_meta.dtype, m_meta.device);
}

void Tensor::set_impl(std::shared_ptr<TensorImpl> impl, DataType new_dtype) {
    m_impl = std::move(impl);
    m_meta.dtype = new_dtype;
    // The new impl is a fresh contiguous buffer; any view metadata from the
    // previous storage no longer applies.
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
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Zeros(shape_, dtype, device);
}
Tensor Tensor::Zeros(std::vector<int64_t> shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Zeros(shape, dtype, device);
}
Tensor Tensor::Ones(std::initializer_list<int64_t> shape, DataType dtype, Device device) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Ones(shape_, dtype, device);
}
Tensor Tensor::Ones(std::vector<int64_t> shape, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Ones(shape, dtype, device);
}
Tensor Tensor::Fill(std::initializer_list<int64_t> shape, float value, DataType dtype, Device device) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Fill(shape_, dtype, value, device);
}
Tensor Tensor::Fill(std::vector<int64_t> shape, float value, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Fill(shape, dtype, value, device);
}
Tensor Tensor::Random(std::initializer_list<int64_t> shape, float min, float max, DataType dtype, Device device) {
    if (shape.size() == 0) throw std::runtime_error("Tensor shape cannot be empty");
    std::vector<int64_t> shape_(shape);
    return ops::Random(shape_, dtype, min, max, device);
}
Tensor Tensor::Random(std::vector<int64_t> shape, float min, float max, DataType dtype, Device device) {
    if (shape.empty()) throw std::runtime_error("Tensor shape cannot be empty");
    return ops::Random(shape, dtype, min, max, device);
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

Tensor& Tensor::squeeze(int dim) {
    if (dim >= 0 && static_cast<size_t>(dim) >= m_meta.shape.size())
        throw std::runtime_error("squeeze dim out of range");
    if (dim < 0) {
        for (int i = m_meta.shape.size() - 1; i >= 0; i--) {
            if (m_meta.shape[i] == 1) {
                m_meta.shape.erase(m_meta.shape.begin() + i);
            }
        }
    }
    if (m_meta.shape[dim] == 1) m_meta.shape.erase(m_meta.shape.begin() + dim);
    m_meta.calculate_strides();
    return *this;
}

Tensor& Tensor::unsqueeze(size_t dim) {
    if (dim > m_meta.shape.size()) throw std::runtime_error("unsqueeze dim out of range");
    m_meta.shape.insert(m_meta.shape.begin() + dim, 1);
    m_meta.calculate_strides();  // 原实现漏了重算，shape/strides 会长度不匹配
    return *this;
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

template int8_t Tensor::at<int8_t>(std::initializer_list<int64_t> idxs);
template int16_t Tensor::at<int16_t>(std::initializer_list<int64_t> idxs);
template int32_t Tensor::at<int32_t>(std::initializer_list<int64_t> idxs);
template int64_t Tensor::at<int64_t>(std::initializer_list<int64_t> idxs);
template float16 Tensor::at<float16>(std::initializer_list<int64_t> idxs);
template bfloat16 Tensor::at<bfloat16>(std::initializer_list<int64_t> idxs);
template float32 Tensor::at<float32>(std::initializer_list<int64_t> idxs);
template float64 Tensor::at<float64>(std::initializer_list<int64_t> idxs);

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
