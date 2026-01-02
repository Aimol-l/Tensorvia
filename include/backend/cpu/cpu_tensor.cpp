#include "cpu_tensor.h"
#include <print>
using namespace via;

size_t CPUTensor::numel() const{return m_numel;}
std::shared_ptr<ContextImpl> CPUTensor::context() const{
    return nullptr;
}
void CPUTensor::init(void *ptr,size_t numel, DataType dtype){
    m_dtype = dtype;
    m_numel = numel;
    size_t total_bytes = m_numel * calc_dtype_size(m_dtype);
    void* raw_ptr = std::malloc(total_bytes);
    if (!raw_ptr) 
        throw std::bad_alloc();
    if(ptr != nullptr) std::memcpy(raw_ptr,ptr,total_bytes);
    m_data.reset(raw_ptr);
}
CPUTensor::CPUTensor(size_t numel, DataType dtype){
    this->init(nullptr,numel,dtype);
}
CPUTensor::CPUTensor(void *ptr,size_t numel, DataType dtype){
    this->init(ptr,numel,dtype);
}

std::unique_ptr<TensorImpl> CPUTensor::clone() const{
    auto cloned = std::make_unique<CPUTensor>(this->m_numel, this->m_dtype);
    size_t bytes = this->m_numel * calc_dtype_size(this->m_dtype);
    std::memcpy(cloned->m_data.get(), this->m_data.get(), bytes);
    return cloned;
}
std::unique_ptr<TensorImpl> CPUTensor::clone_as_contiguous(const Metadata& meta) const{
    auto cloned = std::make_unique<CPUTensor>(meta.numel, this->m_dtype);

    size_t dtype_size = calc_dtype_size(this->m_dtype);
    size_t total_elements = meta.numel;
    if (total_elements == 0) return cloned;
    const char* src_base = static_cast<const char*>(this->m_data.get()) 
                         + meta.offset * dtype_size;
    char* dst = static_cast<char*>(cloned->m_data.get());
    // 预计算 divisors for coordinate decomposition
    std::vector<size_t> divisors(meta.shape.size());
    if (!meta.shape.empty()) {
        divisors[0] = total_elements / meta.shape[0];
        for (size_t i = 1; i < meta.shape.size(); ++i) {
            divisors[i] = divisors[i - 1] / meta.shape[i];
        }
    }
    for (size_t dst_idx = 0; dst_idx < total_elements; ++dst_idx) {
        size_t src_elem_offset = 0;
        size_t temp = dst_idx;
        for (size_t dim = 0; dim < meta.shape.size(); ++dim) {
            size_t coord = (dim == 0) ? temp / divisors[0] 
                                      : (temp % divisors[dim - 1]) / divisors[dim];
            src_elem_offset += coord * meta.strides[dim];
            if (dim == 0) temp = temp % divisors[0];
        }
        std::memcpy(dst + dst_idx * dtype_size,
                    src_base + src_elem_offset * dtype_size,
                    dtype_size);
    }
    return cloned;
}
// cpu --> cpu
void CPUTensor::copy_to(std::shared_ptr<TensorImpl> dst) const{
    if (!dst->data()) throw std::runtime_error("Destination pointer is null");
    if (!m_data) throw std::runtime_error("Source data is null");
    size_t bytes = m_numel * calc_dtype_size(m_dtype);
    std::memcpy(dst->data(), m_data.get(), bytes);
}