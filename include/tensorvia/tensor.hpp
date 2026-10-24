#pragma once

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

#include "tensorvia/types.hpp"
#include "tensorvia/context.hpp"

struct Metadata {
    size_t numel = 0;
    int64_t offset = 0;
    Device device;
    DataType dtype;
    DimArray shape;
    DimArray strides;
    // 连续性缓存：strides 变更点（calculate_strides/_make_view）重算，
    // 避免每次 is_contiguous() 都逐维比较。
    bool is_contig = true;

    void calculate_strides() {
        strides.resize(shape.size());
        if (shape.empty()) {
            update_contiguous();
            return;
        }
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        update_contiguous();
    }

    void update_contiguous() {
        int64_t expected = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            if (strides[i] != expected) {
                is_contig = false;
                return;
            }
            expected *= shape[i];
        }
        is_contig = true;
    }
};

class TensorImpl {
public:
    virtual ~TensorImpl() = default;
    virtual void* data() = 0;
    virtual size_t numel() const = 0;
    virtual const void* data() const = 0;
    virtual void copy_to(std::shared_ptr<TensorImpl> dst) const = 0;
    virtual std::unique_ptr<TensorImpl> clone() const = 0;
    virtual std::unique_ptr<TensorImpl> clone_as_contiguous(const Metadata&) const = 0;
    virtual std::shared_ptr<ContextImpl> context() const = 0;

    // Host <-> device transfer primitives. The default implementation treats the
    // implementation as host memory (CPUTensor relies on this); backends such as
    // Vulkan override both to stage through host-visible buffers.
    virtual void copy_to_host(void* dst, DataType dtype) const {
        std::memcpy(dst, data(), numel() * calc_dtype_size(dtype));
    }
    virtual void copy_from_host(const void* src, DataType dtype) {
        std::memcpy(data(), src, numel() * calc_dtype_size(dtype));
    }
};

class Tensor {
private:
    Metadata m_meta;
    std::shared_ptr<TensorImpl> m_impl;

    Tensor(std::shared_ptr<TensorImpl> impl, Metadata meta)
        : m_meta(std::move(meta)), m_impl(std::move(impl)) {}
    Tensor _make_view(DimArray shape,
                      DimArray strides,
                      int64_t offset) const;
    // View metadata (offset/strides) is only valid for the shared buffer it was
    // created from. Before handing storage to a different backend, collapse a
    // strided/offset view into a standalone contiguous buffer.
    void materialize_for_transfer();

public:
    void* data();
    const void* data() const;

    Tensor clone() const;
    Device device() const;
    DataType dtype() const;

    Tensor to_type(DataType) const;  // returns a new tensor, leaves *this untouched
    void to_type_(DataType);         // in-place conversion

    void to_host();
    void to_device(Device target);
    void to_device(uint32_t id = 0);  // legacy: moves CPU tensor to the accelerator

    int64_t offset() const { return m_meta.offset; }

    int64_t strides(int i) const;
    std::span<const int64_t> strides() const;

    Tensor& to_contiguous();
    Tensor contiguous() const;
    bool is_contiguous() const;

    size_t dims() const;
    size_t numel() const { return m_meta.numel; }
    Tensor empty_like(Tensor& tensor) const;

    std::shared_ptr<TensorImpl> get_impl() const;
    void set_impl(std::shared_ptr<TensorImpl> impl, DataType new_dtype);

    Tensor view(std::vector<int64_t> new_shape_list);
    Tensor view(std::initializer_list<int64_t> new_shape);

    Tensor();
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor(std::vector<int64_t> shape, DataType dtype);
    Tensor(std::initializer_list<int64_t> shape, DataType dtype);
    Tensor(std::vector<int64_t> shape, DataType dtype, Device device);
    Tensor(std::initializer_list<int64_t> shape, DataType dtype, Device device);
    // span 重载：内部以 DimArray/span 传 shape 时避免先落 vector
    Tensor(std::span<const int64_t> shape, DataType dtype);
    Tensor(std::span<const int64_t> shape, DataType dtype, Device device);
    Tensor(void* ptr, std::vector<int64_t> shape, DataType dtype, Device device);
    Tensor(void* ptr, std::initializer_list<int64_t> shape, DataType dtype, Device device);

    template <typename T>
    Tensor(std::vector<T>& vec, std::initializer_list<int64_t> shape);
    template <typename T>
    Tensor(std::vector<T>& vec, std::vector<int64_t> shape);

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // 复用 Tensor 对象自身的所有内部容量（shape vector + 数据 buffer）。
    // 用于热循环里反复使用同一个 Tensor 容器，避免反复构造/析构。
    // - shape/dtype 与当前一致时：零分配
    // - dtype 一致、numel 变化：复用 shape vector 容量，buffer 走 pool 重分配
    // - dtype 或 device 变化：全量重分配（仍走 pool，比从头构造快）
    // 调用后 buffer 内容未初始化（垃圾值）。
    void reset(std::vector<int64_t> shape, DataType dtype);
    void reset(std::initializer_list<int64_t> shape, DataType dtype);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    friend Tensor operator+(float a, const Tensor& other);
    Tensor operator+(float a) const;
    friend Tensor operator-(float a, const Tensor& other);
    Tensor operator-(float a) const;
    friend Tensor operator*(float a, const Tensor& other);
    Tensor operator*(float a) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(float a) const;

    int64_t shape(int i);
    int64_t shape(int i) const;
    std::span<const int64_t> shape() const;

    Tensor t();
    Tensor permute(const std::vector<int64_t>& dims) const;
    Tensor permute(std::initializer_list<int64_t> dims) const;
    Tensor slice(const std::vector<std::pair<int64_t, int64_t>>& ranges) const;

    void reshape(std::vector<int64_t>& newshape);
    void reshape(std::initializer_list<int64_t> newshape);

    Tensor& squeeze(int dim = -1);
    Tensor& unsqueeze(size_t dim);

    Tensor operator==(const Tensor& other) const;
    Tensor operator==(const float val) const;
    Tensor operator!=(const Tensor& other) const;
    Tensor operator!=(const float val) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator>(const float val) const;
    Tensor operator>=(const Tensor& other) const;
    Tensor operator>=(const float val) const;
    Tensor operator<(const Tensor& other) const;
    Tensor operator<(const float val) const;
    Tensor operator<=(const Tensor& other) const;
    Tensor operator<=(const float val) const;

    template <typename T>
    T at(std::initializer_list<int64_t> idxs);

    template <typename T>
    T operator[](std::initializer_list<int64_t> idxs);

    static Tensor Zeros(std::initializer_list<int64_t> shape,
                        DataType dtype = DataType::FLOAT32,
                        Device device = Device::CPU);
    static Tensor Zeros(std::vector<int64_t> shape,
                        DataType dtype = DataType::FLOAT32,
                        Device device = Device::CPU);
    static Tensor Ones(std::initializer_list<int64_t> shape,
                       DataType dtype = DataType::FLOAT32,
                       Device device = Device::CPU);
    static Tensor Ones(std::vector<int64_t> shape,
                       DataType dtype = DataType::FLOAT32,
                       Device device = Device::CPU);
    static Tensor Fill(std::initializer_list<int64_t> shape, float value,
                       DataType dtype = DataType::FLOAT32,
                       Device device = Device::CPU);
    static Tensor Fill(std::vector<int64_t> shape, float value,
                       DataType dtype = DataType::FLOAT32,
                       Device device = Device::CPU);
    static Tensor Random(std::initializer_list<int64_t> shape, float min = 0.0f, float max = 1.0f,
                         DataType dtype = DataType::FLOAT32,
                         Device device = Device::CPU);
    static Tensor Random(std::vector<int64_t> shape, float min = 0.0f, float max = 1.0f,
                         DataType dtype = DataType::FLOAT32,
                         Device device = Device::CPU);
};
