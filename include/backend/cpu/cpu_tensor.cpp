#include "cpu_tensor.h"
#include <print>

size_t CPUTensor::numel() const{return m_numel;}
std::shared_ptr<ContextImpl> CPUTensor::context() const{
    return nullptr;
}
void CPUTensor::reshape(std::vector<int64_t> &newshape){
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
}

void CPUTensor::init(void *ptr, std::vector<int64_t> shape, DataType dtype){
    m_dtype = dtype;
    m_numel = calc_numel(shape);
    m_shape = shape;
    size_t total_bytes = m_numel * calc_dtype_size(m_dtype);
    void* raw_ptr = std::malloc(total_bytes);
    if (!raw_ptr) 
        throw std::bad_alloc();
    if(ptr != nullptr) std::memcpy(raw_ptr,ptr,total_bytes);
    m_data.reset(raw_ptr);
}
void CPUTensor::reshape(std::initializer_list<int64_t> newshape){
    m_shape.clear();
    m_shape.assign(newshape.begin(), newshape.end());
}
CPUTensor::CPUTensor(std::vector<int64_t> shape, DataType dtype){
    this->init(nullptr,shape,dtype);
}
CPUTensor::CPUTensor(void *ptr, std::vector<int64_t> shape, DataType dtype){
    this->init(ptr,shape,dtype);
}

std::unique_ptr<TensorImpl> CPUTensor::clone() const{
    auto cloned = std::make_unique<CPUTensor>(this->m_shape, this->m_dtype);
    // 计算字节大小
    size_t bytes = this->m_numel * calc_dtype_size(this->m_dtype);
    // 拷贝数据
    std::memcpy(cloned->m_data.get(), this->m_data.get(), bytes);
    return cloned;
}
// cpu --> cpu
void CPUTensor::copy_to(void* dst) const{
    if (!dst) throw std::runtime_error("Destination pointer is null");
    if (!m_data) throw std::runtime_error("Source data is null");
    size_t bytes = m_numel * calc_dtype_size(m_dtype);
    std::memcpy(dst, m_data.get(), bytes);
}