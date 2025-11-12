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

class Tensor {
private:
    size_t m_numel;
    Device m_device;
    DataType m_dtype;
    std::vector<int64_t> m_shape;
    std::shared_ptr<TensorImpl> m_impl;
public:
    void*data(){
        return m_impl->data();
    }
    const void* data() const{
        return m_impl->data();
    }
    void to_host(); // 若使用cuda|sycl|vulkan后端，可同步回cpu
    Tensor clone() const;
    Device device() const;
    DataType dtype() const;
    Tensor to_type(DataType);
    void to_device(uint32_t id=0); // 若当前使用cpu,可以移动到device

    size_t dims()const;
    size_t numel() const {return m_numel;}
    Tensor empty_like(Tensor& tensor) const;

    // 构造函数
    Tensor();
    Tensor(const Tensor& other);            // 拷贝构造函数
    Tensor(Tensor&& other) noexcept;        // 移动构造函数
    Tensor(std::vector<int64_t> shape,DataType dtype);
    Tensor(std::initializer_list<int64_t> shape,DataType dtype);
    Tensor(std::vector<int64_t> shape,DataType dtype,Device device);
    Tensor(std::initializer_list<int64_t> shape,DataType dtype,Device device);
    // 从外部指针创建张量
    Tensor(void* ptr, std::vector<int64_t> shape,DataType dtype,Device device);
    Tensor(void* ptr,std::initializer_list<int64_t> shape,DataType dtype,Device device);// 深拷贝ptr的内容

    // 从stl容器创建张量,vector
    template<typename T>
    Tensor(std::vector<T>& vec,std::initializer_list<int64_t> shape);

    template<typename T>
    Tensor(std::vector<T>& vec, std::vector<int64_t> shape);

    Tensor& operator=(const Tensor& other); // 拷贝赋值运算符
    Tensor& operator=(Tensor&& other) noexcept;  // 移动赋值运算符

    // 切片
    Tensor slice(const std::vector<std::pair<int,int>>& ranges) const;

    // 加法
    Tensor add(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    // 减法
    Tensor sub(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    // 元素乘法
    Tensor dot(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    // 实现数与张量的交换率
    friend Tensor operator+(float a, const Tensor& other);
    Tensor operator+(float a) const;
    friend Tensor operator-(float a, const Tensor& other);
    Tensor operator-(float a) const;
    friend Tensor operator*(float a, const Tensor& other);
    Tensor operator*(float a) const;
    // 元素除法
    Tensor div(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    // 矩阵除数不具备交换律
    Tensor operator/(float a) const;
    // 矩阵乘法
    Tensor matmul(const Tensor& other) const;
    
    // 支持负数下标的访问：int64_t val = shape(-1);
    int64_t shape(int i);
    int64_t shape(int i)const;
    std::vector<int64_t> shape()const;
    // reshape,不改变数据排列
    void reshape(std::vector<int64_t>& newshape);
    void reshape(std::initializer_list<int64_t> newshape);

    std::shared_ptr<TensorImpl> get_impl() const;

    Tensor& squeeze(int dim = -1); // 清除shape中的1
    Tensor& unsqueeze(size_t dim); // 在shape指定维度dim中插入1

    float sum();
    Tensor sum(int axis);
    float mean();
    Tensor mean(int axis);
    float max();
    Tensor max(int axis);
    float min();
    Tensor min(int axis);

    bool all(float val);
    bool any(float val);
    size_t nonzero();
    Tensor argmax(int axis);
    Tensor argmin(int axis);

    // 比较方法
    Tensor equal(const Tensor& other) const;
    Tensor not_equal(const Tensor& other) const;
    Tensor greater(const Tensor& other) const;
    Tensor greater_equal(const Tensor& other) const;
    Tensor less(const Tensor& other) const;
    Tensor less_equal(const Tensor& other) const;

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

