#pragma once
#include <memory>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>
#include <array>
#include "core/context.h"
#include "core/type_traits.h"

class TensorImpl {
public:
    virtual ~TensorImpl() = default;
    virtual void* data() = 0; 
    virtual size_t numel() const = 0;
    virtual void copy_to(void* dst) const = 0;                      // 同设备
    virtual const void* data() const = 0;
    virtual void reshape(std::vector<int64_t>& newshape)=0;
    virtual void reshape(std::initializer_list<int64_t> newshape)=0;
    virtual std::unique_ptr<TensorImpl> clone() const = 0;
    virtual std::shared_ptr<ContextImpl> context() const = 0;
};
struct Metadata {
    bool contiguous;              // 记录当前tensor是否为连续存储
    size_t numel;                 // 记录tensor元素个数
    Device device;                // 记录tensor数据所在后端设备
    DataType dtype;               // 记录元素数据类型
    std::vector<int64_t> shape;   // shape
    std::vector<int64_t> strides; // 每维步长（单位：元素个数）
    size_t storage_offset = 0;    // 起始偏移（单位：元素个数）
};

class Tensor {
private:
    Metadata m_meta;
    std::shared_ptr<TensorImpl> m_impl;
public:
    void*data();
    const void* data() const;

    Tensor clone() const;
    Device device() const;
    DataType dtype() const;
    Tensor to_type(DataType);

    void to_host();
    void to_device(uint32_t id=0);

    size_t dims()const;
    size_t numel() const {return m_meta.numel;}
    Tensor empty_like(Tensor& tensor) const;
    std::shared_ptr<TensorImpl> get_impl() const;

    // 构造函数
    Tensor();
    Tensor(const Tensor& other);            // 拷贝构造函数
    Tensor(Tensor&& other) noexcept;        // 移动构造函数
    Tensor(std::vector<int64_t> shape,DataType dtype);
    Tensor(std::initializer_list<int64_t> shape,DataType dtype);
    Tensor(std::vector<int64_t> shape,DataType dtype,Device device);
    Tensor(std::initializer_list<int64_t> shape,DataType dtype,Device device);
    Tensor(void* ptr, std::vector<int64_t> shape,DataType dtype,Device device);
    Tensor(void* ptr,std::initializer_list<int64_t> shape,DataType dtype,Device device);// 深拷贝ptr的内容
    template<typename T>
    Tensor(std::vector<T>& vec,std::initializer_list<int64_t> shape);
    template<typename T>
    Tensor(std::vector<T>& vec, std::vector<int64_t> shape);

    Tensor& operator=(const Tensor& other); // 拷贝赋值运算符
    Tensor& operator=(Tensor&& other) noexcept;  // 移动赋值运算符

    // 切片

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
    int64_t shape(int i)const;
    std::vector<int64_t> shape()const;

    // reshape,不改变数据排列
    void reshape(std::vector<int64_t>& newshape);
    void reshape(std::initializer_list<int64_t> newshape);

    Tensor& squeeze(int dim = -1); // 清除shape中的1
    Tensor& unsqueeze(size_t dim); // 在shape指定维度dim中插入1
    Tensor slice(const std::vector<std::pair<int,int>>& ranges) const;

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
    
    // 元素访问
    template<typename T>
    T at(std::initializer_list<int64_t> idxs);

    template<typename T>
    T operator[](std::initializer_list<int64_t> idxs);

    // 常用静态函数
    static Tensor Zeros(std::initializer_list<int64_t> shape, DataType dtype = DataType::FLOAT32);
    static Tensor Zeros(std::vector<int64_t> shape, DataType dtype = DataType::FLOAT32);
    static Tensor Ones(std::initializer_list<int64_t> shape, DataType dtype = DataType::FLOAT32);
    static Tensor Ones(std::vector<int64_t> shape, DataType dtype = DataType::FLOAT32);
    static Tensor Fill(std::initializer_list<int64_t> shape, float value, DataType dtype = DataType::FLOAT32);
    static Tensor Fill(std::vector<int64_t> shape, float value, DataType dtype);
    static Tensor Random(std::initializer_list<int64_t> shape, float min = 0.0f, float max = 1.0f, DataType dtype = DataType::FLOAT32);
    static Tensor Random(std::vector<int64_t> shape, float min = 0.0f, float max = 1.0f, DataType dtype = DataType::FLOAT32);
};

