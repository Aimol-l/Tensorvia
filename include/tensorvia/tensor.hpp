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
    // storage 落在该后端的哪个实例上。视图与 clone 必须继承它；
    // 其余一切从它推导自己的 context。
    int device_index = 0;
    DataType dtype;
    DimArray shape;
    DimArray strides;
    // 连续性缓存：strides 变更点（calculate_strides/_make_view）重算，
    // 避免每次 is_contiguous() 都逐维比较。
    bool is_contig = true;

    DeviceId device_id() const { return {device, device_index}; }

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

    // 宿主 <-> 设备传输原语。默认实现把 impl 当作宿主内存（CPUTensor 依赖这一点）；
    // Vulkan 等设备后端重写这两个函数，经 host-visible buffer 中转。
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
    // 在 `id` 指定的实例上新建一个连续张量——不像公开构造函数那样把裸 `Device`
    // 解析成线程 current device。核心路径（clone/contiguous/empty_like）不能做那种
    // 解析：结果必须落在源张量所在的实例上。
    static Tensor allocate(DimArray shape, DataType dtype, DeviceId id);
    // 视图元数据（offset/strides）只对创建它的那块共享 buffer 有效。把 storage
    // 交给另一个后端之前，先把跨步/带 offset 的视图物化成独立的连续 buffer。
    void materialize_for_transfer();

public:
    void* data();
    const void* data() const;

    Tensor clone() const;
    Device device() const;
    int device_index() const;
    DeviceId device_id() const;
    DataType dtype() const;

    Tensor to_type(DataType) const;  // 返回新张量，*this 不变
    void to_type_(DataType);         // 原地转换

    // 显式原地入口，沿用 PyTorch 的尾随下划线命名（FR-6）。它们改写共享 storage，
    // 所有别名到这块存储的句柄都能看到写入。写入只走连续 buffer：非连续视图会被
    // 拒绝，而不是乱序写入。
    Tensor& fill_(float value);
    Tensor& zero_();
    Tensor& copy_(const Tensor& src);
    Tensor& add_(float value);
    Tensor& sub_(float value);
    Tensor& mul_(float value);
    Tensor& div_(float value);
    Tensor& abs_();
    Tensor& relu_();
    Tensor& clamp_(float min, float max);

    void to_host();
    // 把本张量搬到另一个设备实例。非 CPU -> 非 CPU 经宿主内存中转：先保证正确性，
    // 而且这种路径很少见。已经在 `id` 上的张量原样不动。
    void to_device(DeviceId id);
    // `index` < 0 表示 `target` 后端上调用线程的 current device。
    void to_device(Device target, int index = -1);

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
    // 替换底层存储。不变式：impl 的 context 必须就是本张量 device_index 命名的那个
    // ——算子入口的守卫按该索引从注册表路由后续结果，而不是经过 impl。
    void set_impl(std::shared_ptr<TensorImpl> impl, DataType new_dtype);

    // 零拷贝视图。它们都共享 storage，并保持 offset 与 strides 一致；
    // 谁都不会写源张量（FR-4）。
    Tensor view(std::vector<int64_t> new_shape_list) const;
    Tensor view(std::initializer_list<int64_t> new_shape) const;
    // Reshape 返回新张量：连续时是视图，否则是连续拷贝（PyTorch 的规则）。
    // -1 表示推导该维长度。
    Tensor reshape(std::vector<int64_t> newshape) const;
    Tensor reshape(std::initializer_list<int64_t> newshape) const;
    // `squeeze(dim)` 只去掉长度为 1 的维；其它长度下是原样返回的视图。
    // `squeeze()` 去掉所有长度为 1 的维。
    Tensor squeeze() const;
    Tensor squeeze(int64_t dim) const;
    Tensor unsqueeze(int64_t dim) const;
    // narrow(dim, start, length) 与 select(dim, index) 都是单轴切片。
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor select(int64_t dim, int64_t index) const;
    // expand：把长度为 1 的维以 0 步长扩到任意长度；其余维必须已经匹配。
    // `shape` 右对齐。
    Tensor expand(std::vector<int64_t> shape) const;
    Tensor expand(std::initializer_list<int64_t> shape) const;
    Tensor expand_as(const Tensor& other) const;

    Tensor();
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor(std::vector<int64_t> shape, DataType dtype);
    Tensor(std::initializer_list<int64_t> shape, DataType dtype);
    // dtype 与 device 一起传入（FR-7）；device 默认取 default_device()。
    explicit Tensor(std::vector<int64_t> shape, const TensorOptions& options = {});
    Tensor(std::initializer_list<int64_t> shape, const TensorOptions& options);
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

    // 单元素张量的值，拷回宿主。numel() != 1 会被拒绝。设备后端上这是一次同步拷贝：
    // 宿主/设备边界就在这里跨越，绝不隐式发生。
    template <typename T>
    T item() const;

    // 按本句柄的视图元数据写一个元素，因此在视图上写会穿透到它别名的 storage。
    // BOOL 写入规范化为 0/1。与 at() 一样仅支持 CPU。
    template <typename T>
    void set(std::initializer_list<int64_t> idxs, T value);

    template <typename T>
    T operator[](std::initializer_list<int64_t> idxs);

    static Tensor Zeros(std::initializer_list<int64_t> shape,
                        DataType dtype = default_dtype(),
                        Device device = Device::CPU);
    static Tensor Zeros(std::vector<int64_t> shape,
                        DataType dtype = default_dtype(),
                        Device device = Device::CPU);
    static Tensor Ones(std::initializer_list<int64_t> shape,
                       DataType dtype = default_dtype(),
                       Device device = Device::CPU);
    static Tensor Ones(std::vector<int64_t> shape,
                       DataType dtype = default_dtype(),
                       Device device = Device::CPU);
    static Tensor Fill(std::initializer_list<int64_t> shape, float value,
                       DataType dtype = default_dtype(),
                       Device device = Device::CPU);
    static Tensor Fill(std::vector<int64_t> shape, float value,
                       DataType dtype = default_dtype(),
                       Device device = Device::CPU);
    static Tensor Random(std::initializer_list<int64_t> shape, float min = 0.0f, float max = 1.0f,
                         DataType dtype = default_dtype(),
                         Device device = Device::CPU);
    static Tensor Random(std::vector<int64_t> shape, float min = 0.0f, float max = 1.0f,
                         DataType dtype = default_dtype(),
                         Device device = Device::CPU);

    // TensorOptions 形式（FR-7）：dtype 与设备实例一次传入。工厂落在哪个设备由
    // TensorOptions 解析，未指定索引时取线程 current device。
    static Tensor empty(std::vector<int64_t> shape, const TensorOptions& options = {});
    static Tensor empty(std::initializer_list<int64_t> shape, const TensorOptions& options);
    static Tensor Zeros(std::vector<int64_t> shape, const TensorOptions& options);
    static Tensor Zeros(std::initializer_list<int64_t> shape, const TensorOptions& options);
    static Tensor Ones(std::vector<int64_t> shape, const TensorOptions& options);
    static Tensor Ones(std::initializer_list<int64_t> shape, const TensorOptions& options);
    static Tensor Fill(std::vector<int64_t> shape, float value, const TensorOptions& options);
    static Tensor Fill(std::initializer_list<int64_t> shape, float value, const TensorOptions& options);
    static Tensor Random(std::vector<int64_t> shape, float min, float max, const TensorOptions& options);
    static Tensor Random(std::initializer_list<int64_t> shape, float min, float max,
                         const TensorOptions& options);
};
